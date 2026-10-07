//! 日文文本分析：文本 -> OpenJTalk 全上下文标签。
//!
//! 音素和韵律符号由 App/Frontend/frontend.js 从标签里取，这里只负责分词、读音和重音。

use std::ffi::{c_char, CStr, CString};
use std::sync::OnceLock;

use jpreprocess::kind::JPreprocessDictionaryKind;
use jpreprocess::{DefaultTokenizer, JPreprocess, SystemDictionaryConfig};

static JTALK: OnceLock<Option<JPreprocess<DefaultTokenizer>>> = OnceLock::new();

fn jtalk() -> Option<&'static JPreprocess<DefaultTokenizer>> {
    JTALK
        .get_or_init(|| {
            SystemDictionaryConfig::Bundled(JPreprocessDictionaryKind::NaistJdic)
                .load()
                .ok()
                .map(|system| JPreprocess::with_dictionaries(system, None))
        })
        .as_ref()
}

/// 返回每行一个标签；词典载入失败或分析出错时返回 None。
pub fn ja_labels(text: &str) -> Option<String> {
    let labels = jtalk()?.extract_fullcontext(text).ok()?;
    let lines: Vec<String> = labels.iter().map(|label| label.to_string()).collect();
    Some(lines.join("\n"))
}

/// C 接口。返回的字符串要用 `gsv_free` 释放；失败时返回空指针。
///
/// # Safety
/// `text` 必须是以 NUL 结尾的 UTF-8 字符串，或者空指针。
#[no_mangle]
pub unsafe extern "C" fn gsv_ja_labels(text: *const c_char) -> *mut c_char {
    if text.is_null() {
        return std::ptr::null_mut();
    }
    let Ok(text) = CStr::from_ptr(text).to_str() else {
        return std::ptr::null_mut();
    };
    match ja_labels(text).and_then(|labels| CString::new(labels).ok()) {
        Some(labels) => labels.into_raw(),
        None => std::ptr::null_mut(),
    }
}

/// # Safety
/// `ptr` 必须是 `gsv_ja_labels` 返回的指针，且只能释放一次。
#[no_mangle]
pub unsafe extern "C" fn gsv_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}
