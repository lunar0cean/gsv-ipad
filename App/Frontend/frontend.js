// 文本 -> 音素编号。移植自 GSV-TTS-Lite 的文本处理（gsv_tts/TextProcessor.py 和 GPT_SoVITS/G2P/）。
// 同一份代码在电脑上用 Node 对照原版测试，在 iPad 上由 JavaScriptCore 运行。
//
// 宿主要提供：
//   GSV_DATA            data.js 里的符号表
//   __jaLabels(text)    日文片段 -> OpenJTalk 全上下文标签，每行一个（iPad 上由 Rust 库提供）
//   GSV_ZH              zh.js 里的中文处理（可选，没有时中文输入会报错）
var GSV = (function () {
  "use strict";

  var SYMBOL_ID = {};
  GSV_DATA.symbols.forEach(function (symbol, index) {
    SYMBOL_ID[symbol] = index;
  });

  // Symbols.punctuation
  var PUNCTUATION = ["!", "?", "…", ",", ".", "-"];
  var CONSECUTIVE_PUNCTUATION = /([!?…,.\-])([!?…,.\-])+/g;

  // ---------- 日文（G2P/Japanese/japanese.py）----------

  var JA_CHARS = "A-Za-z\\d\\u3005\\u3040-\\u30ff\\u4e00-\\u9fff\\uff11-\\uff19\\uff21-\\uff3a\\uff41-\\uff5a\\uff66-\\uff9d";
  var JA_MARK = new RegExp("[^" + JA_CHARS + "]", "u");
  var JA_MARK_ALL = new RegExp("[^" + JA_CHARS + "]", "gu");
  var JA_REP_MAP = {
    "：": ",", "；": ",", "，": ",", "。": ".", "！": "!", "？": "?",
    "\n": ".", "·": ",", "、": ",", "...": "…"
  };

  function jaNormalize(text) {
    // 引号是排版符号，不发音
    return text.replace(/[「」『』]/g, "").replace(CONSECUTIVE_PUNCTUATION, "$1");
  }

  function numericFeature(regex, label) {
    var match = regex.exec(label || "");
    return match ? parseInt(match[1], 10) : -50;
  }

  // 从全上下文标签里取音素，并按重音信息插入韵律符号：[ 升调、] 降调、# 重音短语边界
  function jaProsody(segment) {
    var labels = __jaLabels(segment).split("\n").filter(function (line) { return line.length > 0; });
    var count = labels.length;
    var phones = [];
    for (var n = 0; n < count; n++) {
      var label = labels[n];
      var p3 = /-(.*?)\+/.exec(label)[1];
      if ("AEIOU".indexOf(p3) >= 0) {
        p3 = p3.toLowerCase();  // 清化元音按普通元音处理
      }
      if (p3 === "sil") {
        if (n === 0) {
          phones.push("^");
        } else if (n === count - 1) {
          var e3 = numericFeature(/!(\d+)_/, label);
          if (e3 === 0) {
            phones.push("$");
          } else if (e3 === 1) {
            phones.push("?");
          }
        }
        continue;
      }
      if (p3 === "pau") {
        phones.push("_");
        continue;
      }
      phones.push(p3);

      var a1 = numericFeature(/\/A:([0-9\-]+)\+/, label);
      var a2 = numericFeature(/\+(\d+)\+/, label);
      var a3 = numericFeature(/\+(\d+)\//, label);
      var f1 = numericFeature(/\/F:(\d+)_/, label);
      var a2Next = numericFeature(/\+(\d+)\+/, labels[n + 1]);
      if (a3 === 1 && a2Next === 1 && "aeiouAEIOUNcl".indexOf(p3) >= 0) {
        phones.push("#");
      } else if (a1 === 0 && a2Next === a2 + 1 && a2 !== f1) {
        phones.push("]");
      } else if (a2 === 1 && a2Next === 2) {
        phones.push("[");
      }
    }
    return phones;
  }

  function jaG2P(normText) {
    var text = normText.replace(/％/g, "パーセント").toLowerCase();
    var sentences = text.split(JA_MARK);
    var marks = text.match(JA_MARK_ALL) || [];
    var phones = [];
    for (var i = 0; i < sentences.length; i++) {
      if (sentences[i].length > 0) {
        var prosody = jaProsody(sentences[i]);
        phones = phones.concat(prosody.slice(1, prosody.length - 1));
      }
      if (i < marks.length && marks[i] !== " ") {
        phones.push(marks[i]);
      }
    }
    return phones.map(function (ph) {
      return Object.prototype.hasOwnProperty.call(JA_REP_MAP, ph) ? JA_REP_MAP[ph] : ph;
    });
  }

  // ---------- 入口（TextProcessor.get_phones_and_bert，语种由调用方指定）----------

  function g2p(text, lang) {
    text = text.replace(/ {2,}/g, " ");
    var norm, phones;
    if (lang === "ja") {
      norm = jaNormalize(text);
      phones = jaG2P(norm);
    } else if (lang === "zh") {
      if (typeof GSV_ZH === "undefined") {
        throw new Error("这个版本还不支持中文输入");
      }
      norm = GSV_ZH.normalize(text);
      phones = GSV_ZH.g2p(norm);
    } else {
      throw new Error("不支持的语种：" + lang);
    }
    phones = phones.map(function (ph) {
      return Object.prototype.hasOwnProperty.call(SYMBOL_ID, ph) ? ph : "UNK";
    });
    return {
      norm: norm,
      phones: phones,
      ids: phones.map(function (ph) { return SYMBOL_ID[ph]; })
    };
  }

  // ---------- 切句（TextProcessor.cut_text，参数取网页界面的默认值）----------

  var SENTENCE_END = [".", "。", "?", "？", "!", "！", ",", "，", ":", "：", ";", "；", "、"];
  var CUT_MIN_LENGTH = 10;
  var CUT_MUTE_SECONDS = 0.2;
  var CUT_MUTE_SCALE = {
    ".": 1.5, "。": 1.5, "?": 1.5, "？": 1.5, "!": 1.5, "！": 1.5, ",": 0.8, "，": 0.8, "、": 0.6
  };

  function isSpace(token) { return /^\s+$/u.test(token); }
  function isDigits(token) { return /^\p{Nd}+$/u.test(token); }
  function isCutMark(token) { return SENTENCE_END.indexOf(token) >= 0; }

  function cutText(text, minLength) {
    text = text.replace(/^\n+|\n+$/g, "");
    var tokens = text.match(/[a-zA-Z0-9]+|[^\n]/gu) || [];
    var merged = [];
    var items = [];
    var logicalCount = 0;
    for (var i = 0; i < tokens.length; i++) {
      var token = tokens[i];
      items.push(token);
      if (!isCutMark(token) && !isSpace(token)) {
        logicalCount += 1;
      }
      if (isCutMark(token)) {
        var isDecimal = token === "." && i > 0 && i < tokens.length - 1 &&
          isDigits(tokens[i - 1]) && isDigits(tokens[i + 1]);
        if (!isDecimal && logicalCount >= minLength) {
          merged.push(items.join(""));
          items = [];
          logicalCount = 0;
        }
      }
    }
    if (items.length > 0) {
      merged.push(items.join(""));
    }
    return merged.filter(function (item) {
      return Array.from(item).some(function (ch) { return !isCutMark(ch) && !isSpace(ch); });
    });
  }

  // 引号不发音，两种语言的规范化最后都会把它们去掉。必须在切句之前就去掉：
  // 否则像「……。」这样的一行，末尾的「」」会被当成没有标点收尾的内容，切出一个读不出声音的片段
  var QUOTES = /[「」『』“”‘’"'＂＇]/g;
  var SILENT_PHONES = PUNCTUATION.concat(["UNK", "SP", "SP2", "SP3", "_"]);

  function isSpeakable(phones) {
    return phones.some(function (ph) { return SILENT_PHONES.indexOf(ph) < 0; });
  }

  // 整段文字 -> 一组可以逐个合成的片段。每个片段前面加一个句号，减少「漏读很短的第一句」
  function prepare(text, lang) {
    var segments = [];
    text.split(/\n+/).forEach(function (line) {
      line = line.replace(QUOTES, "").trim();
      if (line.length === 0) {
        return;
      }
      if (!isCutMark(line.charAt(line.length - 1))) {
        line += ".";
      }
      cutText(line, CUT_MIN_LENGTH).forEach(function (cut) {
        var result = g2p("。" + cut, lang);
        if (!isSpeakable(result.phones)) {
          return;  // 只有标点或符号的片段，模型什么都生成不出来，不送去合成
        }
        var last = cut.charAt(cut.length - 1);
        var scale = Object.prototype.hasOwnProperty.call(CUT_MUTE_SCALE, last) ? CUT_MUTE_SCALE[last] : 1.0;
        segments.push({ text: cut, ids: result.ids, pause: CUT_MUTE_SECONDS * scale });
      });
    });
    return segments;
  }

  // 给 Swift 调用：出错时返回 {"error": "..."}，不抛异常
  function prepareJSON(text, lang) {
    try {
      return JSON.stringify({ segments: prepare(text, lang) });
    } catch (error) {
      return JSON.stringify({ error: String(error && error.message ? error.message : error) });
    }
  }

  return {
    PUNCTUATION: PUNCTUATION,
    SYMBOL_ID: SYMBOL_ID,
    g2p: g2p,
    cutText: cutText,
    prepare: prepare,
    prepareJSON: prepareJSON
  };
})();

if (typeof module !== "undefined" && module.exports) {
  module.exports = GSV;
}
