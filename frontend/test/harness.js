// 在 Node 里按 iPad 上的方式加载文本处理脚本：先数据，后逻辑，全部放进同一个全局环境。
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const FRONTEND_DIR = path.join(__dirname, "..", "..", "App", "Frontend");
const SCRIPT_ORDER = ["data.js", "zh.js", "frontend.js"];

function loadFrontend(jaLabels) {
  const context = {
    console,
    __jaLabels: jaLabels || (() => ""),
    // 与 iPad 上的行为一致：文件不存在时返回空字符串
    __loadText: (name) => {
      const file = path.join(FRONTEND_DIR, name);
      return fs.existsSync(file) ? fs.readFileSync(file, "utf8") : "";
    },
  };
  vm.createContext(context);
  for (const name of SCRIPT_ORDER) {
    const file = path.join(FRONTEND_DIR, name);
    if (fs.existsSync(file)) {
      vm.runInContext(fs.readFileSync(file, "utf8"), context, { filename: name });
    }
  }
  return {
    context,
    GSV: vm.runInContext("GSV", context),
    GSV_ZH: vm.runInContext("typeof GSV_ZH === 'undefined' ? null : GSV_ZH", context),
  };
}

module.exports = { loadFrontend, FRONTEND_DIR };
