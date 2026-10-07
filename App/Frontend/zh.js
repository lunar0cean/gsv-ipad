// 中文文本 -> 音素。移植自 GSV-TTS-Lite 的 G2P/Chinese（chinese.py、tone_sandhi.py、Normalization/），
// 以及它依赖的 jieba 分词（含词性标注）和 pypinyin 的词语切分。目标是与原版逐音素一致。
//
// 宿主要提供 __loadText(name)：读取同目录下的词典数据文件。数据在第一次处理中文时才载入。
var GSV_ZH = (function () {
  "use strict";

  var MIN_FLOAT = -3.14e100;
  var hasOwn = Object.prototype.hasOwnProperty;
  function has(object, key) { return hasOwn.call(object, key); }

  // ---------- 数据 ----------

  var D = null;

  function data() {
    if (D) {
      return D;
    }
    var misc = JSON.parse(__loadText("zh_misc.json"));
    var pinyin = JSON.parse(__loadText("zh_pinyin.json"));
    var hmm = JSON.parse(__loadText("zh_hmm.json"));

    // jieba 的词典：词 -> 词频。每个词的所有前缀也要登记（词频记 0），分词时靠它判断能否继续往后匹配
    var freq = new Map();
    var tags = new Map();
    var total = 0;
    var lines = __loadText("zh_dict.txt").split("\n");
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i].trim();
      if (!line) {
        continue;
      }
      var parts = line.split(" ");
      var word = parts[0];
      var count = parseInt(parts[1], 10);
      freq.set(word, count);
      tags.set(word, parts[2]);
      total += count;
      for (var c = 1; c < word.length; c++) {
        var fragment = word.slice(0, c);
        if (!freq.has(fragment)) {
          freq.set(fragment, 0);
        }
      }
    }

    // pypinyin 的词语前缀集合，最大正向匹配用
    var prefixes = new Set();
    Object.keys(pinyin.phrases).concat(pinyin.prefix_only).forEach(function (phrase) {
      for (var n = 1; n <= phrase.length; n++) {
        prefixes.add(phrase.slice(0, n));
      }
    });

    var numeric = new Set(Array.from(misc.numeric));
    D = {
      freq: freq, tags: tags, logTotal: Math.log(total),
      chars: pinyin.chars, phrases: pinyin.phrases, prefixes: prefixes,
      finalHMM: hmm.final, posHMM: hmm.pos, posStates: Object.keys(hmm.pos.trans),
      opencpop: misc.opencpop, numeric: numeric
    };
    return D;
  }

  // ---------- 数字读法（Normalization/num.py）----------

  var DIGITS = "零一二三四五六七八九";
  var UNIT_POWERS = [8, 4, 3, 2, 1];
  var UNITS = { 1: "十", 2: "百", 3: "千", 4: "万", 8: "亿" };

  function stripLeadingZeros(text) { return text.replace(/^0+/, ""); }
  function stripTrailingZeros(text) { return text.replace(/0+$/, ""); }

  function getValue(value, useZero) {
    var stripped = stripLeadingZeros(value);
    if (stripped.length === 0) {
      return [];
    }
    if (stripped.length === 1) {
      if (useZero && stripped.length < value.length) {
        return [DIGITS[0], DIGITS[+stripped]];
      }
      return [DIGITS[+stripped]];
    }
    var largest = 1;
    for (var i = 0; i < UNIT_POWERS.length; i++) {
      if (UNIT_POWERS[i] < stripped.length) {
        largest = UNIT_POWERS[i];
        break;
      }
    }
    var first = value.slice(0, value.length - largest);
    var second = value.slice(value.length - largest);
    return getValue(first, true).concat([UNITS[largest]], getValue(second, true));
  }

  function verbalizeCardinal(value) {
    if (!value) {
      return "";
    }
    value = stripLeadingZeros(value);
    if (value.length === 0) {
      return DIGITS[0];
    }
    var symbols = getValue(value, true);
    // 「一十几」读作「十几」
    if (symbols.length >= 2 && symbols[0] === DIGITS[1] && symbols[1] === UNITS[1]) {
      symbols = symbols.slice(1);
    }
    return symbols.join("");
  }

  function verbalizeDigit(value, altOne) {
    var result = Array.from(value).map(function (digit) { return DIGITS[+digit]; }).join("");
    return altOne ? result.split("一").join("幺") : result;
  }

  function num2str(value) {
    var parts = value.split(".");
    var integer = parts[0];
    var decimal = parts.length === 2 ? parts[1] : "";
    var result = verbalizeCardinal(integer);
    decimal = /0$/.test(decimal) ? stripTrailingZeros(decimal) + "0" : stripTrailingZeros(decimal);
    if (decimal) {
      result = (result ? result : "零") + "点" + verbalizeDigit(decimal, false);
    }
    return result;
  }

  function timeNum2str(value) {
    var result = num2str(stripLeadingZeros(value));
    return value.charAt(0) === "0" ? DIGITS[0] + result : result;
  }

  // ---------- 文本规范化（Normalization/text_normlization.py 等）----------

  var SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ";
  var POWER_MAP = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "ˣ": "x", "ʸ": "y", "ⁿ": "n"
  };
  var ASMD_MAP = { "+": "加", "-": "减", "×": "乘", "÷": "除", "=": "等于" };
  var COM_QUANTIFIERS = "(处|台|架|枚|趟|幅|平|方|堵|间|床|株|批|项|例|列|篇|栋|注|亩|封|艘|把|目|套|段|人|所|朵|匹|张|座|回|场|尾|条|个|首|阙|阵|网|炮|顶|丘|棵|只|支|袭|辆|挑|担|颗|壳|窠|曲|墙|群|腔|砣|座|客|贯|扎|捆|刀|令|打|手|罗|坡|山|岭|江|溪|钟|队|单|双|对|出|口|头|脚|板|跳|枝|件|贴|针|线|管|名|位|身|堂|课|本|页|家|户|层|丝|毫|厘|分|钱|两|斤|担|铢|石|钧|锱|忽|(千|毫|微)克|毫|厘|(公)分|分|寸|尺|丈|里|寻|常|铺|程|(千|分|厘|毫|微)米|米|撮|勺|合|升|斗|石|盘|碗|碟|叠|桶|笼|盆|盒|杯|钟|斛|锅|簋|篮|盘|桶|罐|瓶|壶|卮|盏|箩|箱|煲|啖|袋|钵|年|月|日|季|刻|时|周|天|秒|分|小时|旬|纪|岁|世|更|夜|春|夏|秋|冬|代|伏|辈|丸|泡|粒|颗|幢|堆|条|根|支|道|面|片|张|颗|块|元|(亿|千万|百万|万|千|百)|(亿|千万|百万|万|千|百|美|)元|(亿|千万|百万|万|千|百|十|)吨|(亿|千万|百万|万|千|百|)块|角|毛|分)";
  var MEASURES = [
    ["cm2", "平方厘米"], ["cm²", "平方厘米"], ["cm3", "立方厘米"], ["cm³", "立方厘米"], ["cm", "厘米"],
    ["db", "分贝"], ["ds", "毫秒"], ["kg", "千克"], ["km", "千米"], ["m2", "平方米"], ["m²", "平方米"],
    ["m³", "立方米"], ["m3", "立方米"], ["ml", "毫升"], ["m", "米"], ["mm", "毫米"], ["s", "秒"]
  ];
  var UNIT_ALT = "%|°C|℃|度|摄氏度|cm2|cm²|cm3|cm³|cm|db|ds|kg|km|m2|m²|m³|m3|ml|m|mm|s";
  var NUMBER_ALT = "((-?)((\\d+)(\\.\\d+)?)|(\\.(\\d+)))";
  var ASMD_OPERAND = "((-?)((\\d+)(\\.\\d+)?[" + SUP + "]*)|(\\.\\d+[" + SUP + "]*)|([A-Za-z][" + SUP + "]*))";
  var ASMD_SOURCE = ASMD_OPERAND + "([\\+\\-\\×÷=])" + ASMD_OPERAND;

  var RE_DATE = /(\d{4}|\d{2})年((0?[1-9]|1[0-2])月)?(((0?[1-9])|((1|2)[0-9])|30|31)([日号]))?/g;
  var RE_DATE2 = /(\d{4})([- /.])(0[1-9]|1[012])\2(0[1-9]|[12][0-9]|3[01])/g;
  var RE_TIME = /([0-1]?[0-9]|2[0-3]):([0-5][0-9])(:([0-5][0-9]))?/g;
  var RE_TIME_RANGE = /([0-1]?[0-9]|2[0-3]):([0-5][0-9])(:([0-5][0-9]))?(~|-)([0-1]?[0-9]|2[0-3]):([0-5][0-9])(:([0-5][0-9]))?/g;
  var RE_TO_RANGE = new RegExp(NUMBER_ALT + "(" + UNIT_ALT + ")[~]" + NUMBER_ALT + "(" + UNIT_ALT + ")", "g");
  var RE_TEMPERATURE = /(-?)(\d+(\.\d+)?)(°C|℃|度|摄氏度)/g;
  var RE_ASMD = new RegExp(ASMD_SOURCE, "g");
  var RE_ASMD_TEST = new RegExp(ASMD_SOURCE);
  var RE_POWER = new RegExp("[" + SUP + "]+", "g");
  var RE_FRAC = /(-?)(\d+)\/(\d+)/g;
  var RE_PERCENTAGE = /(-?)(\d+(\.\d+)?)%/g;
  var RE_MOBILE_PHONE = /(?<!\d)((\+?86 ?)?1([38]\d|5[0-35-9]|7[678]|9[89])\d{8})(?!\d)/g;
  var RE_TELEPHONE = /(?<!\d)((0(10|2[1-3]|[3-9]\d{2})-?)?[1-9]\d{6,7})(?!\d)/g;
  var RE_NATIONAL_UNIFORM_NUMBER = /(400)(-)?\d{3}(-)?\d{4}/g;
  var RE_RANGE = /(?<![\d\+\-\×÷=])((-?)((\d+)(\.\d+)?))[-~]((-?)((\d+)(\.\d+)?))(?![\d\+\-\×÷=])/g;
  var RE_INTEGER = /(-)(\d+)/g;
  var RE_VERSION_NUM = /((\d+)(\.\d+)(\.\d+)?(\.\d+)+)/g;
  var RE_DECIMAL_NUM = /(-?)((\d+)(\.\d+))|(\.(\d+))/g;
  var RE_POSITIVE_QUANTIFIERS = new RegExp("(\\d+)([多余几\\+])?" + COM_QUANTIFIERS, "g");
  var RE_DEFAULT_NUM = /\d{3}\d*/g;
  var RE_NUMBER = /(-?)((\d+)(\.\d+)?)|(\.(\d+))/g;

  function replaceAll(text, from, to) { return text.split(from).join(to); }

  function replaceTime(isRange) {
    return function () {
      var g = arguments;
      var hour = g[1], minute = g[2], second = g[4];
      var result = num2str(hour) + "点";
      if (stripLeadingZeros(minute)) {
        result += parseInt(minute, 10) === 30 ? "半" : timeNum2str(minute) + "分";
      }
      if (second && stripLeadingZeros(second)) {
        result += timeNum2str(second) + "秒";
      }
      if (isRange) {
        var hour2 = g[6], minute2 = g[7], second2 = g[9];
        result += "至" + num2str(hour2) + "点";
        if (stripLeadingZeros(minute2)) {
          // 原版这里判断的是前一个时刻的分钟数，照搬
          result += parseInt(minute, 10) === 30 ? "半" : timeNum2str(minute2) + "分";
        }
        if (second2 && stripLeadingZeros(second2)) {
          result += timeNum2str(second2) + "秒";
        }
      }
      return result;
    };
  }

  function replaceNumber() {
    var g = arguments;
    if (g[5]) {
      return num2str(g[5]);
    }
    return (g[1] ? "负" : "") + num2str(g[2]);
  }

  function phone2str(phone, mobile) {
    var parts = mobile ? phone.replace(/^\++|\++$/g, "").split(/\s+/).filter(Boolean) : phone.split("-");
    return parts.map(function (part) { return verbalizeDigit(part, true); }).join("，");
  }

  function postReplace(sentence) {
    var pairs = [
      ["/", "每"], ["①", "一"], ["②", "二"], ["③", "三"], ["④", "四"], ["⑤", "五"], ["⑥", "六"], ["⑦", "七"],
      ["⑧", "八"], ["⑨", "九"], ["⑩", "十"], ["α", "阿尔法"], ["β", "贝塔"], ["γ", "伽玛"], ["Γ", "伽玛"],
      ["δ", "德尔塔"], ["Δ", "德尔塔"], ["ε", "艾普西龙"], ["ζ", "捷塔"], ["η", "依塔"], ["θ", "西塔"],
      ["Θ", "西塔"], ["ι", "艾欧塔"], ["κ", "喀帕"], ["λ", "拉姆达"], ["Λ", "拉姆达"], ["μ", "缪"], ["ν", "拗"],
      ["ξ", "克西"], ["Ξ", "克西"], ["ο", "欧米克伦"], ["π", "派"], ["Π", "派"], ["ρ", "肉"], ["ς", "西格玛"],
      ["Σ", "西格玛"], ["σ", "西格玛"], ["τ", "套"], ["υ", "宇普西龙"], ["φ", "服艾"], ["Φ", "服艾"], ["χ", "器"],
      ["ψ", "普赛"], ["Ψ", "普赛"], ["ω", "欧米伽"], ["Ω", "欧米伽"],
      ["+", "加"], ["-", "减"], ["×", "乘"], ["÷", "除"], ["=", "等"]
    ];
    pairs.forEach(function (pair) { sentence = replaceAll(sentence, pair[0], pair[1]); });
    return sentence.replace(/[-——《》【】<=>{}()（）#&@“”^_|\\]/g, "");
  }

  function normalizeSentence(sentence) {
    // 全角字母、数字、空格转半角
    sentence = sentence.replace(/[Ａ-Ｚａ-ｚ０-９]/g, function (ch) {
      return String.fromCharCode(ch.charCodeAt(0) - 65248);
    }).replace(/　/g, " ");

    sentence = sentence.replace(RE_DATE, function () {
      var g = arguments;
      var result = "";
      if (g[1]) { result += verbalizeDigit(g[1], false) + "年"; }
      if (g[3]) { result += verbalizeCardinal(g[3]) + "月"; }
      if (g[5]) { result += verbalizeCardinal(g[5]) + g[9]; }
      return result;
    });
    sentence = sentence.replace(RE_DATE2, function () {
      var g = arguments;
      return verbalizeDigit(g[1], false) + "年" + verbalizeCardinal(g[3]) + "月" + verbalizeCardinal(g[4]) + "日";
    });
    sentence = sentence.replace(RE_TIME_RANGE, replaceTime(true));
    sentence = sentence.replace(RE_TIME, replaceTime(false));
    sentence = sentence.replace(RE_TO_RANGE, function (match) { return replaceAll(match, "~", "至"); });
    sentence = sentence.replace(RE_TEMPERATURE, function () {
      var g = arguments;
      // 原版取单位时取错了分组，结果单位总是读「度」，照搬
      return (g[1] ? "零下" : "") + num2str(g[2]) + "度";
    });
    MEASURES.forEach(function (pair) { sentence = replaceAll(sentence, pair[0], pair[1]); });

    while (RE_ASMD_TEST.test(sentence)) {
      sentence = sentence.replace(RE_ASMD, function () {
        var g = arguments;
        return g[1] + ASMD_MAP[g[8]] + g[9];
      });
    }
    sentence = sentence.replace(RE_POWER, function (match) {
      return "的" + Array.from(match).map(function (ch) { return POWER_MAP[ch]; }).join("") + "次方";
    });
    sentence = sentence.replace(RE_FRAC, function () {
      var g = arguments;
      return (g[1] ? "负" : "") + num2str(g[3]) + "分之" + num2str(g[2]);
    });
    sentence = sentence.replace(RE_PERCENTAGE, function () {
      var g = arguments;
      return (g[1] ? "负" : "") + "百分之" + num2str(g[2]);
    });
    sentence = sentence.replace(RE_MOBILE_PHONE, function (match) { return phone2str(match, true); });
    sentence = sentence.replace(RE_TELEPHONE, function (match) { return phone2str(match, false); });
    sentence = sentence.replace(RE_NATIONAL_UNIFORM_NUMBER, function (match) { return phone2str(match, false); });
    sentence = sentence.replace(RE_RANGE, function () {
      var g = arguments;
      return g[1].replace(RE_NUMBER, replaceNumber) + "到" + g[6].replace(RE_NUMBER, replaceNumber);
    });
    sentence = sentence.replace(RE_INTEGER, function () {
      var g = arguments;
      return "负" + num2str(g[2]);
    });
    sentence = sentence.replace(RE_VERSION_NUM, function (match) {
      return Array.from(match).map(function (ch) { return ch === "." ? "点" : num2str(ch); }).join("");
    });
    sentence = sentence.replace(RE_DECIMAL_NUM, replaceNumber);
    sentence = sentence.replace(RE_POSITIVE_QUANTIFIERS, function () {
      var g = arguments;
      var number = num2str(g[1]);
      if (number === "二") {
        number = "两";
      }
      var extra = g[2] === "+" ? "多" : (g[2] ? g[2] : "");
      return number + extra + g[3];
    });
    sentence = sentence.replace(RE_DEFAULT_NUM, function (match) { return verbalizeDigit(match, true); });
    sentence = sentence.replace(RE_NUMBER, replaceNumber);
    return postReplace(sentence);
  }

  function splitSentences(text) {
    text = replaceAll(text, " ", "");
    text = text.replace(/[——《》【】<>{}()（）#&@“”^_|\\]/g, "");
    text = text.replace(/([：、，；。？！,;?!][”’]?)/g, "$1\n");
    return text.trim().split(/\n+/).map(function (sentence) { return sentence.trim(); });
  }

  var REP_MAP = {
    "：": ",", "；": ",", "，": ",", "。": ".", "！": "!", "？": "?", "\n": ".", "·": ",", "、": ",",
    "...": "…", "$": ".", "/": ",", "—": "-", "~": "…", "～": "…"
  };
  var RE_REP = /：|；|，|。|！|？|\n|·|、|\.\.\.|\$|\/|—|~|～/g;

  function replacePunctuation(text) {
    text = replaceAll(replaceAll(text, "嗯", "恩"), "呣", "母");
    text = text.replace(RE_REP, function (match) { return REP_MAP[match]; });
    return text.replace(/[^一-龥!?…,.\-]+/g, "");
  }

  function normalize(text) {
    var result = "";
    splitSentences(text).forEach(function (sentence) {
      result += replacePunctuation(normalizeSentence(sentence));
    });
    // 避免重复标点引起的参考泄露
    return result.replace(/([!?…,.\-])([!?…,.\-])+/g, "$1");
  }

  // ---------- jieba：分词（jieba/__init__.py、finalseg）----------

  function getDAG(sentence) {
    var freq = data().freq;
    var n = sentence.length;
    var dag = new Array(n);
    for (var k = 0; k < n; k++) {
      var ends = [];
      var i = k;
      var fragment = sentence.charAt(k);
      while (i < n && freq.has(fragment)) {
        if (freq.get(fragment)) {
          ends.push(i);
        }
        i += 1;
        fragment = sentence.slice(k, i + 1);
      }
      if (ends.length === 0) {
        ends.push(k);
      }
      dag[k] = ends;
    }
    return dag;
  }

  // 返回每个位置上最优切分的词尾下标。带词性的分词走这个（jieba 的 Python 写法）：
  // 概率相同时取靠后的词尾，与 Python 对元组取 max 的结果一致
  function calcRoute(sentence, dag) {
    var d = data();
    var n = sentence.length;
    var prob = new Array(n + 1);
    var route = new Array(n);
    prob[n] = 0;
    for (var index = n - 1; index >= 0; index--) {
      var best = -Infinity;
      var bestEnd = -1;
      var ends = dag[index];
      for (var j = 0; j < ends.length; j++) {
        var end = ends[j];
        var count = d.freq.get(sentence.slice(index, end + 1));
        var value = Math.log(count || 1) - d.logTotal + prob[end + 1];
        if (value > best || (value === best && end > bestEnd)) {
          best = value;
          bestEnd = end;
        }
      }
      prob[index] = best;
      route[index] = bestEnd;
    }
    return route;
  }

  // 不带词性的分词用的是 jieba_fast 的 C 实现（_get_DAG_and_calc），和上面的 Python 写法有两处不同：
  // 每个位置最多记 12 个候选词尾；概率相同时取先出现的（靠前的）词尾
  function calcRouteFast(sentence) {
    var d = data();
    var n = sentence.length;
    var prob = new Array(n + 1);
    var route = new Array(n);
    var dag = new Array(n);
    var k, i;
    for (k = 0; k < n; k++) {
      var ends = [];
      i = k;
      var fragment = sentence.charAt(k);
      while (i < n && d.freq.has(fragment) && ends.length < 12) {
        if (d.freq.get(fragment)) {
          ends.push(i);
        }
        i += 1;
        fragment = sentence.slice(k, i + 1);
      }
      if (ends.length === 0) {
        ends.push(k);
      }
      dag[k] = ends;
    }
    prob[n] = 0;
    for (var index = n - 1; index >= 0; index--) {
      var best = -2147483648;
      var bestEnd = index;
      for (i = 0; i < dag[index].length; i++) {
        var end = dag[index][i];
        var count = d.freq.get(sentence.slice(index, end + 1));
        var value = Math.log(count || 1) - d.logTotal + prob[end + 1];
        if (value > best) {
          best = value;
          bestEnd = end;
        }
      }
      prob[index] = best;
      route[index] = bestEnd;
    }
    return route;
  }

  var FINAL_STATES = "BMES";
  var FINAL_PREV = { B: "ES", M: "MB", S: "SE", E: "BM" };

  // finalseg 的 Viterbi，照 jieba_fast 的 C 实现（_viterbi）写，而不是 Python 的参考写法。区别在于：
  // 分数只有严格大于下限 MIN_FLOAT 才算数，否则分数记为 MIN_FLOAT、前一个状态取两个候选里字母大的；
  // 分数相同时取先出现的候选。加法的先后顺序也照搬，保证浮点结果一致
  function finalViterbi(obs) {
    var hmm = data().finalHMM;
    var scores = [{}];
    var back = [{}];
    var t, s, y;
    for (s = 0; s < 4; s++) {
      y = FINAL_STATES[s];
      scores[0][y] = (has(hmm.emit[y], obs[0]) ? hmm.emit[y][obs[0]] : MIN_FLOAT) + hmm.start[y];
    }
    for (t = 1; t < obs.length; t++) {
      scores.push({});
      back.push({});
      for (s = 0; s < 4; s++) {
        y = FINAL_STATES[s];
        var emit = has(hmm.emit[y], obs[t]) ? hmm.emit[y][obs[t]] : MIN_FLOAT;
        var best = MIN_FLOAT;
        var bestState = "";
        var previous = FINAL_PREV[y];
        for (var p = 0; p < 2; p++) {
          var y0 = previous[p];
          var value = emit;
          value += scores[t - 1][y0];
          value += has(hmm.trans[y0], y) ? hmm.trans[y0][y] : MIN_FLOAT;
          if (value > best) {
            best = value;
            bestState = y0;
          }
        }
        if (bestState === "") {
          bestState = previous[0] > previous[1] ? previous[0] : previous[1];
        }
        scores[t][y] = best;
        back[t][y] = bestState;
      }
    }
    var last = obs.length - 1;
    var state = scores[last].S > scores[last].E ? "S" : "E";
    var path = new Array(obs.length);
    for (t = last; t >= 0; t--) {
      path[t] = state;
      state = back[t][state];
    }
    return path;
  }

  function finalCutHan(sentence, out) {
    var path = finalViterbi(sentence);
    var begin = 0;
    var next = 0;
    for (var i = 0; i < sentence.length; i++) {
      if (path[i] === "B") {
        begin = i;
      } else if (path[i] === "E") {
        out.push(sentence.slice(begin, i + 1));
        next = i + 1;
      } else if (path[i] === "S") {
        out.push(sentence.charAt(i));
        next = i + 1;
      }
    }
    if (next < sentence.length) {
      out.push(sentence.slice(next));
    }
  }

  function finalCut(sentence, out) {
    sentence.split(/([一-鿕]+)/).forEach(function (block) {
      if (/^[一-鿕]/.test(block)) {
        finalCutHan(block, out);
      } else {
        block.split(/([a-zA-Z0-9]+(?:\.\d+)?%?)/).forEach(function (piece) {
          if (piece) {
            out.push(piece);
          }
        });
      }
    });
  }

  // 没有被词典覆盖的连续单字：只有一个字就原样输出；整体是词典里的词就拆成单字；否则交给 HMM
  function flushBuffer(buffer, out, single, recognize) {
    if (!buffer) {
      return;
    }
    if (buffer.length === 1) {
      single(buffer);
    } else if (!data().freq.get(buffer)) {
      recognize(buffer);
    } else {
      for (var i = 0; i < buffer.length; i++) {
        single(buffer.charAt(i));
      }
    }
  }

  function cutDAG(sentence, out) {
    var route = calcRouteFast(sentence);
    var push = function (word) { out.push(word); };
    var recognize = function (buffer) { finalCut(buffer, out); };
    var x = 0;
    var buffer = "";
    while (x < sentence.length) {
      var y = route[x] + 1;
      var word = sentence.slice(x, y);
      if (y - x === 1) {
        buffer += word;
      } else {
        flushBuffer(buffer, out, push, recognize);
        buffer = "";
        out.push(word);
      }
      x = y;
    }
    flushBuffer(buffer, out, push, recognize);
  }

  // Tokenizer.cut(sentence, HMM=True)
  function cut(sentence) {
    var out = [];
    sentence.split(/([一-鿕a-zA-Z0-9+#&\._%]+)/).forEach(function (block) {
      if (!block) {
        return;
      }
      if (/^[一-鿕a-zA-Z0-9+#&\._%]/.test(block)) {
        cutDAG(block, out);
      } else {
        block.split(/(\r\n|\s)/).forEach(function (piece) {
          if (/^(\r\n|\s)/.test(piece)) {
            out.push(piece);
          } else {
            for (var i = 0; i < piece.length; i++) {
              out.push(piece.charAt(i));
            }
          }
        });
      }
    });
    return out;
  }

  function cutForSearch(sentence) {
    var freq = data().freq;
    var out = [];
    cut(sentence).forEach(function (word) {
      var i;
      if (word.length > 2) {
        for (i = 0; i < word.length - 1; i++) {
          if (freq.get(word.slice(i, i + 2))) {
            out.push(word.slice(i, i + 2));
          }
        }
      }
      if (word.length > 3) {
        for (i = 0; i < word.length - 2; i++) {
          if (freq.get(word.slice(i, i + 3))) {
            out.push(word.slice(i, i + 3));
          }
        }
      }
      out.push(word);
    });
    return out;
  }

  // ---------- jieba：分词 + 词性（posseg）----------

  // posseg/viterbi.py。状态是「位置|词性」，如 "B|n"；分数相同时取字典序大的状态，与 Python 对元组取 max 一致
  function posViterbi(obs) {
    var d = data();
    var hmm = d.posHMM;
    var scores = [{}];
    var back = [{}];
    var states = has(hmm.char_state, obs[0]) ? hmm.char_state[obs[0]] : d.posStates;
    var i, t, y;
    for (i = 0; i < states.length; i++) {
      y = states[i];
      scores[0][y] = hmm.start[y] + (has(hmm.emit[y], obs[0]) ? hmm.emit[y][obs[0]] : MIN_FLOAT);
      back[0][y] = "";
    }
    for (t = 1; t < obs.length; t++) {
      scores.push({});
      back.push({});
      var previous = Object.keys(back[t - 1]).filter(function (state) {
        return Object.keys(hmm.trans[state]).length > 0;
      });
      var expected = new Set();
      previous.forEach(function (state) {
        Object.keys(hmm.trans[state]).forEach(function (next) { expected.add(next); });
      });
      var candidates = (has(hmm.char_state, obs[t]) ? hmm.char_state[obs[t]] : d.posStates).filter(function (state) {
        return expected.has(state);
      });
      if (candidates.length === 0) {
        candidates = expected.size > 0 ? Array.from(expected) : d.posStates;
      }
      for (i = 0; i < candidates.length; i++) {
        y = candidates[i];
        var emit = has(hmm.emit[y], obs[t]) ? hmm.emit[y][obs[t]] : MIN_FLOAT;
        var best = null;
        var bestState = null;
        for (var p = 0; p < previous.length; p++) {
          var y0 = previous[p];
          var value = scores[t - 1][y0] + (has(hmm.trans[y0], y) ? hmm.trans[y0][y] : -Infinity) + emit;
          if (bestState === null || value > best || (value === best && y0 > bestState)) {
            best = value;
            bestState = y0;
          }
        }
        scores[t][y] = best;
        back[t][y] = bestState;
      }
    }
    var last = obs.length - 1;
    var finalScore = null;
    var state = null;
    Object.keys(back[last]).forEach(function (candidate) {
      var value = scores[last][candidate];
      if (state === null || value > finalScore || (value === finalScore && candidate > state)) {
        finalScore = value;
        state = candidate;
      }
    });
    var route = new Array(obs.length);
    for (t = last; t >= 0; t--) {
      route[t] = state;
      state = back[t][state];
    }
    return route;
  }

  function posCutHan(sentence, out) {
    var route = posViterbi(sentence);
    var begin = 0;
    var next = 0;
    for (var i = 0; i < sentence.length; i++) {
      var position = route[i].charAt(0);
      var tag = route[i].slice(2);
      if (position === "B") {
        begin = i;
      } else if (position === "E") {
        out.push([sentence.slice(begin, i + 1), tag]);
        next = i + 1;
      } else if (position === "S") {
        out.push([sentence.charAt(i), tag]);
        next = i + 1;
      }
    }
    if (next < sentence.length) {
      out.push([sentence.slice(next), route[next].slice(2)]);
    }
  }

  function posCutDetail(sentence, out) {
    sentence.split(/([一-鿕]+)/).forEach(function (block) {
      if (/^[一-鿕]/.test(block)) {
        posCutHan(block, out);
        return;
      }
      block.split(/([\.0-9]+|[a-zA-Z0-9]+)/).forEach(function (piece) {
        if (!piece) {
          return;
        }
        if (/^[\.0-9]+/.test(piece)) {
          out.push([piece, "m"]);
        } else if (/^[a-zA-Z0-9]+/.test(piece)) {
          out.push([piece, "eng"]);
        } else {
          out.push([piece, "x"]);
        }
      });
    });
  }

  function posCutDAG(sentence, out) {
    var tags = data().tags;
    var route = calcRoute(sentence, getDAG(sentence));
    var tagged = function (word) { out.push([word, tags.has(word) ? tags.get(word) : "x"]); };
    var recognize = function (buffer) { posCutDetail(buffer, out); };
    var x = 0;
    var buffer = "";
    while (x < sentence.length) {
      var y = route[x] + 1;
      var word = sentence.slice(x, y);
      if (y - x === 1) {
        buffer += word;
      } else {
        flushBuffer(buffer, out, tagged, recognize);
        buffer = "";
        tagged(word);
      }
      x = y;
    }
    flushBuffer(buffer, out, tagged, recognize);
  }

  // posseg.lcut(sentence)，返回 [[词, 词性], ...]
  function posCut(sentence) {
    var out = [];
    sentence.split(/([一-鿕a-zA-Z0-9+#&\._]+)/).forEach(function (block) {
      if (/^[一-鿕a-zA-Z0-9+#&\._]/.test(block)) {
        posCutDAG(block, out);
        return;
      }
      block.split(/(\r\n|\s)/).forEach(function (piece) {
        if (/^(\r\n|\s)/.test(piece)) {
          out.push([piece, "x"]);
          return;
        }
        var isEnglish = /^[a-zA-Z0-9]+/.test(piece);
        for (var i = 0; i < piece.length; i++) {
          var ch = piece.charAt(i);
          if (/^[\.0-9]+/.test(ch)) {
            out.push([ch, "m"]);
          } else if (isEnglish) {
            out.push([ch, "eng"]);
          } else {
            out.push([ch, "x"]);
          }
        }
      });
    });
    return out;
  }

  // ---------- pypinyin：词语切分和读音 ----------

  function isHan(ch) { return ch >= "一" && ch <= "龥"; }

  // pypinyin.seg.mmseg.Seg.cut（no_non_phrases=True）：最大正向匹配，只认词语库里的词
  function phraseCut(text) {
    var d = data();
    var out = [];
    var remain = text;
    while (remain) {
      var lastValid = "";
      var lastValidIndex = 0;
      var broke = false;
      for (var index = 0; index < remain.length; index++) {
        var word = remain.slice(0, index + 1);
        if (d.prefixes.has(word)) {
          if (has(d.phrases, word)) {
            lastValid = word;
            lastValidIndex = index + 1;
          }
        } else {
          if (lastValid) {
            out.push(lastValid);
            remain = remain.slice(lastValidIndex);
          } else {
            out.push(remain.charAt(0));
            remain = remain.slice(1);
          }
          broke = true;
          break;
        }
      }
      if (!broke) {
        if (lastValid) {
          out.push(lastValid);
          remain = remain.slice(lastValidIndex);
        } else {
          if (has(d.phrases, remain)) {
            out.push(remain);
          } else {
            for (var i = 0; i < remain.length; i++) {
              out.push(remain.charAt(i));
            }
          }
          break;
        }
      }
    }
    return out;
  }

  // lazy_pinyin(word, neutral_tone_with_five=True) 的声母（INITIALS）和带调韵母（FINALS_TONE3）。
  // 没有读音的字符原样返回，连续的非汉字算作一项
  function lazyPinyin(word) {
    var d = data();
    var initials = [];
    var finals = [];
    function pushReading(reading) {
      var bar = reading.indexOf("|");
      initials.push(reading.slice(0, bar));
      finals.push(reading.slice(bar + 1));
    }
    var runs = [];
    for (var i = 0; i < word.length; i++) {
      var ch = word.charAt(i);
      if (runs.length > 0 && isHan(runs[runs.length - 1].charAt(0)) === isHan(ch)) {
        runs[runs.length - 1] += ch;
      } else {
        runs.push(ch);
      }
    }
    runs.forEach(function (run) {
      if (!isHan(run.charAt(0))) {
        initials.push(run);
        finals.push(run);
        return;
      }
      phraseCut(run).forEach(function (piece) {
        if (has(d.phrases, piece)) {
          d.phrases[piece].split(" ").forEach(pushReading);
        } else if (has(d.chars, piece)) {
          pushReading(d.chars[piece]);
        } else {
          initials.push(piece);
          finals.push(piece);
        }
      });
    });
    return { initials: initials, finals: finals };
  }

  // ---------- 变调（tone_sandhi.py）----------

  var MUST_NEURAL = new Set(("麻烦 麻利 鸳鸯 高粱 骨头 骆驼 马虎 首饰 馒头 馄饨 风筝 难为 队伍 阔气 闺女 门道 锄头 铺盖 铃铛 铁匠 钥匙 里脊 里头 部分 那么 道士 造化 迷糊 连累 这么 这个 运气 过去 软和 转悠 踏实 跳蚤 跟头 趔趄 财主 豆腐 讲究 记性 记号 认识 规矩 见识 裁缝 补丁 衣裳 衣服 衙门 街坊 行李 行当 蛤蟆 蘑菇 薄荷 葫芦 葡萄 萝卜 荸荠 苗条 苗头 苍蝇 芝麻 舒服 舒坦 舌头 自在 膏药 脾气 脑袋 脊梁 能耐 胳膊 胭脂 胡萝 胡琴 胡同 聪明 耽误 耽搁 耷拉 耳朵 老爷 老实 老婆 老头 老太 翻腾 罗嗦 罐头 编辑 结实 红火 累赘 糨糊 糊涂 精神 粮食 簸箕 篱笆 算计 算盘 答应 笤帚 笑语 笑话 窟窿 窝囊 窗户 稳当 稀罕 称呼 秧歌 秀气 秀才 福气 祖宗 砚台 码头 石榴 石头 石匠 知识 眼睛 眯缝 眨巴 眉毛 相声 盘算 白净 痢疾 痛快 疟疾 疙瘩 疏忽 畜生 生意 甘蔗 琵琶 琢磨 琉璃 玻璃 玫瑰 玄乎 狐狸 状元 特务 牲口 牙碜 牌楼 爽快 爱人 热闹 烧饼 烟筒 烂糊 点心 炊帚 灯笼 火候 漂亮 滑溜 溜达 温和 清楚 消息 浪头 活泼 比方 正经 欺负 模糊 槟榔 棺材 棒槌 棉花 核桃 栅栏 柴火 架势 枕头 枇杷 机灵 本事 木头 木匠 朋友 月饼 月亮 暖和 明白 时候 新鲜 故事 收拾 收成 提防 挖苦 挑剔 指甲 指头 拾掇 拳头 拨弄 招牌 招呼 抬举 护士 折腾 扫帚 打量 打算 打点 打扮 打听 打发 扎实 扁担 戒指 懒得 意识 意思 情形 悟性 怪物 思量 怎么 念头 念叨 快活 忙活 志气 心思 得罪 张罗 弟兄 开通 应酬 庄稼 干事 帮手 帐篷 希罕 师父 师傅 巴结 巴掌 差事 工夫 岁数 屁股 尾巴 少爷 小气 小伙 将就 对头 对付 寡妇 家伙 客气 实在 官司 学问 学生 字号 嫁妆 媳妇 媒人 婆家 娘家 委屈 姑娘 姐夫 妯娌 妥当 妖精 奴才 女婿 头发 太阳 大爷 大方 大意 大夫 多少 多么 外甥 壮实 地道 地方 在乎 困难 嘴巴 嘱咐 嘟囔 嘀咕 喜欢 喇嘛 喇叭 商量 唾沫 哑巴 哈欠 哆嗦 咳嗽 和尚 告诉 告示 含糊 吓唬 后头 名字 名堂 合同 吆喝 叫唤 口袋 厚道 厉害 千斤 包袱 包涵 匀称 勤快 动静 动弹 功夫 力气 前头 刺猬 刺激 别扭 利落 利索 利害 分析 出息 凑合 凉快 冷战 冤枉 冒失 养活 关系 先生 兄弟 便宜 使唤 佩服 作坊 体面 位置 似的 伙计 休息 什么 人家 亲戚 亲家 交情 云彩 事情 买卖 主意 丫头 丧气 两口 东西 东家 世故 不由 不在 下水 下巴 上头 上司 丈夫 丈人 一辈 那个 菩萨 父亲 母亲 咕噜 邋遢 费用 冤家 甜头 介绍 荒唐 大人 泥鳅 幸福 熟悉 计划 扑腾 蜡烛 姥爷 照顾 喉咙 吉他 弄堂 蚂蚱 凤凰 拖沓 寒碜 糟蹋 倒腾 报复 逻辑 盘缠 喽啰 牢骚 咖喱 扫把 惦记").split(" "));
  var MUST_NOT_NEURAL = new Set(("男子 女子 分子 原子 量子 莲子 石子 瓜子 电子 人人 虎虎 幺幺 干嘛 学子 哈哈 数数 袅袅 局地 以下 娃哈哈 花花草草 留得 耕地 想想 熙熙 攘攘 卵子 死死 冉冉 恳恳 佼佼 吵吵 打打 考考 整整 莘莘 落地 算子 家家户户 青青").split(" "));
  var SANDHI_PUNC = "：，；。？！“”‘’':,;.?!";

  // Python 里对空字符串或空列表取下标会抛异常，原版有几处靠 try/except 接住它，这里保持同样的行为
  function lastOf(sequence) {
    if (sequence === undefined || sequence.length === 0) {
      throw new Error("index out of range");
    }
    return sequence[sequence.length - 1];
  }
  function firstOf(sequence) {
    if (sequence === undefined || sequence.length === 0) {
      throw new Error("index out of range");
    }
    return sequence[0];
  }
  function setTone(finals, index, tone) {
    if (index < 0) {
      index += finals.length;
    }
    if (index < 0 || index >= finals.length) {
      throw new Error("index out of range");
    }
    finals[index] = finals[index].slice(0, -1) + tone;
  }
  function toneOf(final) { return lastOf(final); }
  function tail(word, count) { return word.slice(Math.max(0, word.length - count)); }
  function isNumeric(ch) { return data().numeric.has(ch) || /^\p{N}$/u.test(ch); }
  function isNeuralWord(word) { return MUST_NEURAL.has(word) || MUST_NEURAL.has(tail(word, 2)); }

  function allToneThree(finals) {
    for (var i = 0; i < finals.length; i++) {
      if (toneOf(finals[i]) !== "3") {
        return false;
      }
    }
    return true;
  }

  function splitWord(word) {
    var pieces = cutForSearch(word);
    // 按长度升序的稳定排序，取最短的那个子词
    var first = pieces[0];
    for (var i = 1; i < pieces.length; i++) {
      if (pieces[i].length < first.length) {
        first = pieces[i];
      }
    }
    if (first === undefined) {
      throw new Error("index out of range");
    }
    if (word.indexOf(first) === 0) {
      return [first, word.slice(first.length)];
    }
    return [word.slice(0, word.length - first.length), first];
  }

  function buSandhi(word, finals) {
    if (word.length === 3 && word.charAt(1) === "不") {
      setTone(finals, 1, "5");
    } else {
      for (var i = 0; i < word.length; i++) {
        // 「不」在四声前读二声，如「不怕」
        if (word.charAt(i) === "不" && i + 1 < word.length && toneOf(finals[i + 1]) === "4") {
          setTone(finals, i, "2");
        }
      }
    }
    return finals;
  }

  function yiSandhi(word, finals) {
    var i;
    if (word.indexOf("一") !== -1) {
      var othersNumeric = true;
      for (i = 0; i < word.length; i++) {
        if (word.charAt(i) !== "一" && !isNumeric(word.charAt(i))) {
          othersNumeric = false;
          break;
        }
      }
      if (othersNumeric) {
        return finals;  // 数字串里的「一」，如「一零零」
      }
    }
    if (word.length === 3 && word.charAt(1) === "一" && word.charAt(0) === word.charAt(2)) {
      setTone(finals, 1, "5");  // 「看一看」
    } else if (word.indexOf("第一") === 0) {
      setTone(finals, 1, "1");
    } else {
      for (i = 0; i < word.length; i++) {
        if (word.charAt(i) === "一" && i + 1 < word.length) {
          if (toneOf(finals[i + 1]) === "4") {
            setTone(finals, i, "2");  // 四声前读二声，如「一段」
          } else if (SANDHI_PUNC.indexOf(word.charAt(i + 1)) < 0) {
            setTone(finals, i, "4");  // 其余读四声，如「一天」；后面是标点时仍读一声
          }
        }
      }
    }
    return finals;
  }

  function neuralSandhi(word, pos, finals) {
    var last = word.charAt(word.length - 1);
    // 叠词：奶奶、试试、旺旺
    for (var j = 0; j < word.length; j++) {
      if (j - 1 >= 0 && word.charAt(j) === word.charAt(j - 1) && "nva".indexOf(firstOf(pos)) >= 0 &&
          !MUST_NOT_NEURAL.has(word)) {
        setTone(finals, j, "5");
      }
    }
    var geIndex = word.indexOf("个");
    if (word.length >= 1 && "吧呢哈啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶".indexOf(last) >= 0) {
      setTone(finals, -1, "5");
    } else if (word.length >= 1 && "的地得".indexOf(last) >= 0) {
      setTone(finals, -1, "5");
    } else if (word.length === 1 && "了着过".indexOf(word) >= 0 && (pos === "ul" || pos === "uz" || pos === "ug")) {
      setTone(finals, -1, "5");  // 走了、看着、去过
    } else if (word.length > 1 && "们子".indexOf(last) >= 0 && (pos === "r" || pos === "n") &&
               !MUST_NOT_NEURAL.has(word)) {
      setTone(finals, -1, "5");
    } else if (word.length > 1 && "上下里".indexOf(last) >= 0 && (pos === "s" || pos === "l" || pos === "f")) {
      setTone(finals, -1, "5");  // 桌上、地下、家里
    } else if (word.length > 1 && "来去".indexOf(last) >= 0 &&
               "上下进出回过起开".indexOf(word.charAt(word.length - 2)) >= 0) {
      setTone(finals, -1, "5");  // 上来、下去
    } else if ((geIndex >= 1 && (isNumeric(word.charAt(geIndex - 1)) ||
                                 "几有两半多各整每做是".indexOf(word.charAt(geIndex - 1)) >= 0)) || word === "个") {
      setTone(finals, geIndex, "5");  // 「个」作量词
    } else if (isNeuralWord(word)) {
      setTone(finals, -1, "5");
    }

    var parts = splitWord(word);
    var finalsList = [finals.slice(0, parts[0].length), finals.slice(parts[0].length)];
    for (var i = 0; i < parts.length; i++) {
      if (isNeuralWord(parts[i])) {
        setTone(finalsList[i], -1, "5");
      }
    }
    return finalsList[0].concat(finalsList[1]);
  }

  function threeSandhi(word, finals) {
    if (word.length === 2 && allToneThree(finals)) {
      setTone(finals, 0, "2");
    } else if (word.length === 3) {
      var parts = splitWord(word);
      if (allToneThree(finals)) {
        if (parts[0].length === 2) {
          setTone(finals, 0, "2");  // 双音节 + 单音节，如「蒙古/包」
          setTone(finals, 1, "2");
        } else if (parts[0].length === 1) {
          setTone(finals, 1, "2");  // 单音节 + 双音节，如「纸/老虎」
        }
      } else {
        var finalsList = [finals.slice(0, parts[0].length), finals.slice(parts[0].length)];
        for (var i = 0; i < finalsList.length; i++) {
          var sub = finalsList[i];
          if (allToneThree(sub) && sub.length === 2) {
            setTone(sub, 0, "2");  // 「所有/人」
          } else if (i === 1 && !allToneThree(sub) && toneOf(firstOf(sub)) === "3" &&
                     toneOf(lastOf(finalsList[0])) === "3") {
            setTone(finalsList[0], -1, "2");  // 「好/喜欢」
          }
        }
        finals = finalsList[0].concat(finalsList[1]);
      }
    } else if (word.length === 4) {
      // 四字词按两个双音节词处理
      var halves = [finals.slice(0, 2), finals.slice(2)];
      halves.forEach(function (half) {
        if (allToneThree(half)) {
          setTone(half, 0, "2");
        }
      });
      finals = halves[0].concat(halves[1]);
    }
    return finals;
  }

  function modifiedTone(word, pos, finals) {
    finals = buSandhi(word, finals);
    finals = yiSandhi(word, finals);
    finals = neuralSandhi(word, pos, finals);
    return threeSandhi(word, finals);
  }

  // 「不」和后面的词并在一起，否则它单独成词时变调会出错
  function mergeBu(seg) {
    var out = [];
    var lastWord = "";
    seg.forEach(function (item) {
      var word = item[0];
      if (lastWord === "不") {
        word = lastWord + word;
      }
      if (word !== "不") {
        out.push([word, item[1]]);
      }
      lastWord = word;
    });
    if (lastWord === "不") {
      out.push([lastWord, "d"]);
    }
    return out;
  }

  // 「听/一/听」并成「听一听」；单独的「一」和后面的词并在一起
  function mergeYi(seg) {
    var merged = [];
    var i = 0;
    while (i < seg.length) {
      var word = seg[i][0];
      var done = false;
      if (i - 1 >= 0 && word === "一" && i + 1 < seg.length) {
        var last = merged.length > 0 ? merged[merged.length - 1] : seg[i - 1];
        if (last[0] === seg[i + 1][0] && last[1] === "v" && seg[i + 1][1] === "v") {
          if (merged.length === 0) {
            throw new Error("index out of range");
          }
          merged[merged.length - 1] = [last[0] + "一" + seg[i + 1][0], last[1]];
          i += 2;
          done = true;
        }
      }
      if (!done) {
        merged.push([word, seg[i][1]]);
        i += 1;
      }
    }
    var out = [];
    merged.forEach(function (item) {
      if (out.length > 0 && out[out.length - 1][0] === "一") {
        out[out.length - 1][0] += item[0];
      } else {
        out.push([item[0], item[1]]);
      }
    });
    return out;
  }

  function mergeReduplication(seg) {
    var out = [];
    seg.forEach(function (item) {
      if (out.length > 0 && item[0] === out[out.length - 1][0]) {
        out[out.length - 1][0] += item[0];
      } else {
        out.push([item[0], item[1]]);
      }
    });
    return out;
  }

  function isReduplication(word) { return word.length === 2 && word.charAt(0) === word.charAt(1); }

  // 连续的三声词并在一起，方便后面按三声变调处理。wholeWord 为真时要求两个词都全是三声，
  // 否则只看前一个词的末字和后一个词的首字
  function mergeThreeTones(seg, wholeWord) {
    var subFinals = seg.map(function (item) { return lazyPinyin(item[0]).finals; });
    var mergedLast = seg.map(function () { return false; });
    var out = [];
    for (var i = 0; i < seg.length; i++) {
      var connects = false;
      if (i - 1 >= 0) {
        connects = wholeWord
          ? allToneThree(subFinals[i - 1]) && allToneThree(subFinals[i])
          : toneOf(lastOf(subFinals[i - 1])) === "3" && toneOf(firstOf(subFinals[i])) === "3";
        connects = connects && !mergedLast[i - 1];
      }
      // 前一个词是叠词时不并，叠词要走轻声变调
      if (connects && !isReduplication(seg[i - 1][0]) && seg[i - 1][0].length + seg[i][0].length <= 3) {
        out[out.length - 1][0] += seg[i][0];
        mergedLast[i] = true;
      } else {
        out.push([seg[i][0], seg[i][1]]);
      }
    }
    return out;
  }

  function mergeEr(seg) {
    var out = [];
    for (var i = 0; i < seg.length; i++) {
      if (i - 1 >= 0 && seg[i][0] === "儿" && seg[i - 1][0] !== "#") {
        out[out.length - 1][0] += seg[i][0];
      } else {
        out.push([seg[i][0], seg[i][1]]);
      }
    }
    return out;
  }

  function attempt(seg, merge) {
    try {
      return merge(seg);
    } catch (error) {
      return seg;
    }
  }

  function preMergeForModify(seg) {
    seg = mergeBu(seg);
    seg = attempt(seg, mergeYi);
    seg = mergeReduplication(seg);
    seg = attempt(seg, function (items) { return mergeThreeTones(items, true); });
    seg = attempt(seg, function (items) { return mergeThreeTones(items, false); });
    return mergeEr(seg);
  }

  // ---------- 拼音 -> 音素（chinese.py）----------

  var MUST_ERHUA = new Set(["小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿"]);
  var NOT_ERHUA = new Set(("虐儿 为儿 护儿 瞒儿 救儿 替儿 有儿 一儿 我儿 俺儿 妻儿 拐儿 聋儿 乞儿 患儿 幼儿 孤儿 婴儿 婴幼儿 连体儿 脑瘫儿 流浪儿 体弱儿 混血儿 蜜雪儿 舫儿 祖儿 美儿 应采儿 可儿 侄儿 孙儿 侄孙儿 女儿 男儿 红孩儿 花儿 虫儿 马儿 鸟儿 猪儿 猫儿 狗儿 少儿").split(" "));
  var V_REP = { uei: "ui", iou: "iu", uen: "un" };
  var PINYIN_REP = { ing: "ying", i: "yi", "in": "yin", u: "wu" };
  var SINGLE_REP = { v: "yu", e: "e", i: "y", u: "w" };
  var PUNCTUATION = "!?…,.-";

  function mergeErhua(initials, finals, word, pos) {
    var i;
    for (i = 0; i < finals.length; i++) {
      if (i === finals.length - 1 && word.charAt(i) === "儿" && finals[i] === "er1") {
        finals[i] = "er2";
      }
    }
    if (!MUST_ERHUA.has(word) && (NOT_ERHUA.has(word) || pos === "a" || pos === "j" || pos === "nr")) {
      return [initials, finals];
    }
    if (finals.length !== word.length) {
      return [initials, finals];  // 「……」等情况
    }
    var newInitials = [];
    var newFinals = [];
    for (i = 0; i < finals.length; i++) {
      var final = finals[i];
      if (i === finals.length - 1 && word.charAt(i) === "儿" && (final === "er2" || final === "er5") &&
          !NOT_ERHUA.has(tail(word, 2)) && newFinals.length > 0) {
        final = "er" + toneOf(newFinals[newFinals.length - 1]);  // 与前一个字同调
      }
      newInitials.push(initials[i]);
      newFinals.push(final);
    }
    return [newInitials, newFinals];
  }

  function syllablePhones(initial, final) {
    var opencpop = data().opencpop;
    var tone = final.charAt(final.length - 1);
    var body = final.slice(0, -1);
    var pinyin = initial + body;
    if (initial) {
      if (has(V_REP, body)) {
        pinyin = initial + V_REP[body];
      }
    } else if (has(PINYIN_REP, pinyin)) {
      pinyin = PINYIN_REP[pinyin];
    } else if (pinyin && has(SINGLE_REP, pinyin.charAt(0))) {
      pinyin = SINGLE_REP[pinyin.charAt(0)] + pinyin.slice(1);
    }
    if ("12345".indexOf(tone) < 0 || !has(opencpop, pinyin)) {
      return null;  // 原版在这里会直接报错中断；这里跳过这个读不出来的音节
    }
    var pair = opencpop[pinyin].split(" ");
    return [pair[0], pair[1] + tone];
  }

  // 返回音素，以及规范化文本里每个字对应几个音素（汉字 2 个，标点 1 个）。后者是中文语调模型（BERT）要用的：
  // 它按字给出特征，要按这个数重复成按音素的特征
  function g2pDetail(text) {
    var phones = [];
    var word2ph = [];
    text.split(/(?<=[!?…,.\-])\s*/).forEach(function (segment) {
      if (segment.trim() === "") {
        return;
      }
      segment = segment.replace(/[a-zA-Z]+/g, "");
      var initials = [];
      var finals = [];
      preMergeForModify(posCut(segment)).forEach(function (item) {
        var word = item[0];
        var pos = item[1];
        if (pos === "eng") {
          return;
        }
        var reading = lazyPinyin(word);
        var subFinals = reading.finals;
        try {
          subFinals = modifiedTone(word, pos, reading.finals.slice());
        } catch (error) {
          subFinals = reading.finals;  // 原版在这种情况下会报错中断；这里退回不变调的读法
        }
        var merged = mergeErhua(reading.initials, subFinals, word, pos);
        initials = initials.concat(merged[0]);
        finals = finals.concat(merged[1]);
      });
      for (var i = 0; i < initials.length; i++) {
        if (initials[i] === finals[i]) {
          // 标点。原版只接受单个标点，这里把连在一起的也逐个放进去
          Array.from(initials[i]).forEach(function (ch) {
            if (PUNCTUATION.indexOf(ch) >= 0) {
              phones.push(ch);
              word2ph.push(1);
            } else {
              word2ph.push(0);
            }
          });
          continue;
        }
        var pair = syllablePhones(initials[i], finals[i]);
        if (pair) {
          phones.push(pair[0], pair[1]);
          word2ph.push(2);
        } else {
          word2ph.push(0);
        }
      }
    });
    return { phones: phones, word2ph: word2ph };
  }

  function g2p(text) {
    return g2pDetail(text).phones;
  }

  // ---------- 中文语调模型（BERT）的输入 ----------

  var bertVocabulary;

  // 规范化后的中文只有汉字和几种标点，BERT 分词的结果就是一字一个编号，查不到的记作 [UNK]。
  // 返回 {ids: 含首尾 [CLS]、[SEP] 的编号, repeats: 每个字重复几次}；对不上或没有字表时返回 null
  function bertInput(norm, word2ph) {
    if (bertVocabulary === undefined) {
      var text = __loadText("zh_bert_vocab.json");
      bertVocabulary = text ? JSON.parse(text) : null;
    }
    if (!bertVocabulary || norm.length === 0 || norm.length !== word2ph.length) {
      return null;
    }
    var ids = [bertVocabulary.cls];
    for (var i = 0; i < norm.length; i++) {
      var ch = norm.charAt(i);
      ids.push(has(bertVocabulary.chars, ch) ? bertVocabulary.chars[ch] : bertVocabulary.unk);
    }
    ids.push(bertVocabulary.sep);
    return { ids: ids, repeats: word2ph.slice() };
  }

  return {
    normalize: normalize,
    g2p: g2p,
    g2pDetail: g2pDetail,
    bertInput: bertInput,
    // 以下只给对照测试用
    posCut: posCut,
    cut: cut,
    lazyPinyin: lazyPinyin,
    preMergeForModify: preMergeForModify
  };
})();

if (typeof module !== "undefined" && module.exports) {
  module.exports = GSV_ZH;
}
