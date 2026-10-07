//! 对照测试用的命令行工具：标准输入每行一段日文，输出它的标签，每段后面跟一行 "@@"。

use std::io::{self, BufRead, Write};

fn main() {
    let stdin = io::stdin();
    let mut out = io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line.expect("read stdin");
        let labels = gsv_native::ja_labels(&line).unwrap_or_default();
        writeln!(out, "{labels}\n@@").expect("write stdout");
    }
}
