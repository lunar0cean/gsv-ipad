// native/ 里 Rust 库的 C 接口，作为桥接头文件给 Swift 用。
#ifndef GSV_NATIVE_H
#define GSV_NATIVE_H

// 日文文本 -> OpenJTalk 全上下文标签，每行一个。失败返回 NULL，结果用 gsv_free 释放。
char *gsv_ja_labels(const char *text);
void gsv_free(char *ptr);

#endif
