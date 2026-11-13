//! Test-only helpers that keep a failing assert from dumping a whole body.
//!
//! The helpers compare two values and report the first difference with
//! bounded excerpts. A failing comparison names the comparison, the two
//! lengths and the offset of the first difference, and it prints at most
//! forty bytes of either side and at most eighty bytes of a JSON pointer.
//! A JSON pointer prints escaped, so a key that holds a control character
//! keeps the message on one line. No helper prints a whole request body, a
//! whole response body, a whole rendered template or a whole file body.

use serde_json::Value;
use std::fmt::Debug;

/// The number of bytes of either side that a failing comparison prints.
const EXCERPT_BYTES: usize = 40;

/// The number of bytes of a JSON pointer that a failing comparison prints.
const POINTER_BYTES: usize = 80;

/// Compares two texts and reports the first difference without the texts.
///
/// The call returns when the texts are equal. A mismatch panics with the
/// name, the two lengths in bytes, the offset of the first differing byte
/// and at most forty bytes of either side, cut on a character boundary.
pub fn assert_same_text(name: &str, actual: &str, expected: &str) {
    if actual == expected {
        return;
    }
    let offset = first_difference(actual.as_bytes(), expected.as_bytes());
    panic!(
        "{name}: the texts differ (actual length {} bytes, expected length {} bytes); \
         first difference at byte {offset}: actual {:?}, expected {:?}",
        actual.len(),
        expected.len(),
        text_excerpt(actual, offset),
        text_excerpt(expected, offset),
    );
}

/// Compares two byte buffers and reports the first difference without them.
///
/// The call returns when the buffers are equal. A mismatch panics with the
/// name, the two lengths in bytes, the offset of the first differing byte
/// and at most forty bytes of either side. The bytes print escaped, so a
/// buffer that is not valid UTF-8 prints as it is and the printed form
/// cannot grow by a replacement character.
pub fn assert_same_bytes(name: &str, actual: &[u8], expected: &[u8]) {
    if actual == expected {
        return;
    }
    let offset = first_difference(actual, expected);
    panic!(
        "{name}: the buffers differ (actual length {} bytes, expected length {} bytes); \
         first difference at byte {offset}: actual {}, expected {}",
        actual.len(),
        expected.len(),
        byte_excerpt(actual, offset),
        byte_excerpt(expected, offset),
    );
}

/// Compares two JSON values and reports the first differing field.
///
/// The call returns when the values are equal. A mismatch panics with the
/// name, the JSON pointer of the first differing field and the kind of
/// either value at that pointer. A number, a boolean and a null print as
/// they are, a field that one side lacks prints as `missing`, a string
/// prints its length and at most forty bytes from its first difference,
/// and an array or an object prints its kind and its length only. A long
/// pointer prints its length and its first eighty bytes instead of the
/// whole pointer. A pointer prints escaped, so a key that holds a control
/// character keeps the message on one line.
pub fn assert_same_json(name: &str, actual: &Value, expected: &Value) {
    if actual == expected {
        return;
    }
    let difference = first_json_difference(actual, expected, "");
    let place = if difference.pointer.is_empty() {
        "the document root".to_owned()
    } else {
        bounded_pointer(&difference.pointer)
    };
    panic!(
        "{name}: the JSON values differ at {place}: actual {}, expected {}",
        describe_json_value(difference.actual, difference.expected),
        describe_json_value(difference.expected, difference.actual),
    );
}

/// Compares two buffers of values and reports the first difference.
///
/// The call returns when the buffers hold the same values, which is the
/// equality of `assert_eq!` on two vectors of values. A mismatch panics
/// with the name, the two lengths, the index of the first differing value
/// and the two values at that index, and it prints no other value. The
/// name carries the shape of the compared buffers.
pub fn assert_same_values<T>(name: &str, actual: &[T], expected: &[T])
where
    T: PartialEq + Debug + Copy,
{
    if actual == expected {
        return;
    }
    report_value_difference(name, actual, expected);
}

/// The offset of the first byte where two buffers differ.
///
/// The offset is the length of the shorter buffer when one of them is a
/// prefix of the other.
fn first_difference(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .zip(right)
        .position(|(left, right)| left != right)
        .unwrap_or(left.len().min(right.len()))
}

/// The index of the first value where two buffers of values differ.
///
/// The index is the length of the shorter buffer when one of them is a
/// prefix of the other.
fn first_value_difference<T: PartialEq>(actual: &[T], expected: &[T]) -> usize {
    actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| !actual.eq(expected))
        .unwrap_or(actual.len().min(expected.len()))
}

/// The bytes of `text` around `offset`, cut on a character boundary.
///
/// The window starts at the character boundary at or before the offset and
/// holds at most forty bytes, so a multi-byte character is never split.
fn text_excerpt(text: &str, offset: usize) -> &str {
    let mut start = offset.min(text.len());
    while start > 0 && !text.is_char_boundary(start) {
        start -= 1;
    }
    let mut end = (start + EXCERPT_BYTES).min(text.len());
    while end > start && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[start..end]
}

/// The bytes of `bytes` around `offset`, escaped one byte at a time.
///
/// The window starts at the offset and holds at most forty bytes. Every
/// byte prints escaped, so the printed form stays as long as the window
/// and a byte that is not valid UTF-8 prints as itself.
fn byte_excerpt(bytes: &[u8], offset: usize) -> String {
    let start = offset.min(bytes.len());
    let end = (start + EXCERPT_BYTES).min(bytes.len());
    let mut escaped = String::new();
    for byte in &bytes[start..end] {
        for character in std::ascii::escape_default(*byte) {
            escaped.push(char::from(character));
        }
    }
    escaped
}

/// A bounded description of one value of a buffer of values.
fn describe_value<T: Debug>(value: Option<&T>) -> String {
    match value {
        Some(value) => format!("{value:?}"),
        None => "missing".to_owned(),
    }
}

/// Reports the first difference between two buffers of values.
///
/// The panic names the comparison, the two lengths, the index of the first
/// differing value and the two values at that index.
fn report_value_difference<T>(name: &str, actual: &[T], expected: &[T])
where
    T: PartialEq + Debug + Copy,
{
    let index = first_value_difference(actual, expected);
    panic!(
        "{name}: the values differ (actual length {} values, expected length {} values); \
         first difference at index {index}: actual {}, expected {}",
        actual.len(),
        expected.len(),
        describe_value(actual.get(index)),
        describe_value(expected.get(index)),
    );
}

/// The first field where two JSON values differ.
struct JsonDifference<'a> {
    /// The JSON pointer of the differing field.
    pointer: String,
    /// The value of the field on the actual side, or `None` when the
    /// actual side lacks the field.
    actual: Option<&'a Value>,
    /// The value of the field on the expected side, or `None` when the
    /// expected side lacks the field.
    expected: Option<&'a Value>,
}

/// Walks two JSON values and reports the first field that differs.
///
/// Objects and arrays are walked in their own order, so the pointer names
/// the first field that a reader would meet. A field that one side lacks
/// is a difference of its own.
fn first_json_difference<'a>(
    actual: &'a Value,
    expected: &'a Value,
    pointer: &str,
) -> JsonDifference<'a> {
    match (actual, expected) {
        (Value::Object(left), Value::Object(right)) => {
            for (key, left_value) in left {
                let child = format!("{pointer}/{}", escape_pointer_key(key));
                match right.get(key) {
                    Some(right_value) if left_value != right_value => {
                        return first_json_difference(left_value, right_value, &child);
                    }
                    Some(_) => {}
                    None => {
                        return JsonDifference {
                            pointer: child,
                            actual: Some(left_value),
                            expected: None,
                        };
                    }
                }
            }
            for (key, right_value) in right {
                if !left.contains_key(key) {
                    return JsonDifference {
                        pointer: format!("{pointer}/{}", escape_pointer_key(key)),
                        actual: None,
                        expected: Some(right_value),
                    };
                }
            }
            JsonDifference {
                pointer: pointer.to_owned(),
                actual: Some(actual),
                expected: Some(expected),
            }
        }
        (Value::Array(left), Value::Array(right)) => {
            for (index, left_value) in left.iter().enumerate() {
                let child = format!("{pointer}/{index}");
                match right.get(index) {
                    Some(right_value) if left_value != right_value => {
                        return first_json_difference(left_value, right_value, &child);
                    }
                    Some(_) => {}
                    None => {
                        return JsonDifference {
                            pointer: child,
                            actual: Some(left_value),
                            expected: None,
                        };
                    }
                }
            }
            for (index, right_value) in right.iter().enumerate() {
                if index >= left.len() {
                    return JsonDifference {
                        pointer: format!("{pointer}/{index}"),
                        actual: None,
                        expected: Some(right_value),
                    };
                }
            }
            JsonDifference {
                pointer: pointer.to_owned(),
                actual: Some(actual),
                expected: Some(expected),
            }
        }
        _ => JsonDifference {
            pointer: pointer.to_owned(),
            actual: Some(actual),
            expected: Some(expected),
        },
    }
}

/// Escapes one key of a JSON pointer.
fn escape_pointer_key(key: &str) -> String {
    key.replace('~', "~0").replace('/', "~1")
}

/// A bounded description of a JSON pointer.
///
/// The pointer prints whole when it is short, and it prints escaped, so a
/// key that holds a control character cannot break the message into lines.
/// A long pointer prints its length in bytes and its first eighty bytes,
/// cut on a character boundary, so that a long key cannot fill the message.
fn bounded_pointer(pointer: &str) -> String {
    if pointer.len() <= POINTER_BYTES {
        return format!("{pointer:?}");
    }
    let mut end = POINTER_BYTES;
    while end > 0 && !pointer.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "a pointer of {} bytes, starting with {:?}",
        pointer.len(),
        &pointer[..end]
    )
}

/// A bounded description of one side of a JSON difference.
pub(crate) fn describe_json_value(value: Option<&Value>, other: Option<&Value>) -> String {
    let Some(value) = value else {
        return "missing".to_owned();
    };
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(flag) => flag.to_string(),
        Value::Number(number) => number.to_string(),
        Value::String(text) => {
            let offset = match other {
                Some(Value::String(other_text)) => {
                    first_difference(text.as_bytes(), other_text.as_bytes())
                }
                _ => 0,
            };
            format!(
                "a string of {} bytes: {:?}",
                text.len(),
                text_excerpt(text, offset)
            )
        }
        Value::Array(items) => format!("an array of {} items", items.len()),
        Value::Object(fields) => format!("an object of {} fields", fields.len()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The message of the panic that `body` raises.
    ///
    /// The call catches the panic and reads its payload. The process-wide
    /// panic hook stays as it is, so the capture cannot hide the panic of
    /// another test.
    fn capture_panic(body: impl FnOnce() + std::panic::UnwindSafe) -> String {
        let result = std::panic::catch_unwind(body);
        let payload = result.expect_err("the helper did not panic");
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| {
                payload
                    .downcast_ref::<&str>()
                    .map(|message| (*message).to_owned())
            })
            .expect("the panic carried a message")
    }

    /// The escaped form of a slice of bytes, one byte at a time.
    ///
    /// The helper escapes the bytes the way a comparison does, so a test
    /// can name the exact bytes that the comparison has to print.
    fn escaped_bytes(bytes: &[u8]) -> String {
        let mut escaped = String::new();
        for byte in bytes {
            for character in std::ascii::escape_default(*byte) {
                escaped.push(char::from(character));
            }
        }
        escaped
    }

    #[test]
    fn equal_texts_pass() {
        assert_same_text("the render", "the same text", "the same text");
    }

    #[test]
    fn a_text_mismatch_panics_with_the_first_difference() {
        let message = capture_panic(|| assert_same_text("the render", "abc", "abd"));
        assert_eq!(
            message,
            "the render: the texts differ (actual length 3 bytes, expected length 3 bytes); \
             first difference at byte 2: actual \"c\", expected \"d\""
        );
    }

    #[test]
    fn the_text_message_keeps_the_excerpts_short() {
        let actual = format!("{}b{}", "a".repeat(100), "c".repeat(100));
        let expected = format!("{}d{}", "a".repeat(100), "c".repeat(100));
        let message = capture_panic(|| assert_same_text("the render", &actual, &expected));
        assert!(
            message.contains("first difference at byte 100"),
            "{message}"
        );
        assert!(
            message.contains(&format!("actual {:?}", format!("b{}", "c".repeat(39)))),
            "{message}"
        );
        assert!(
            message.contains(&format!("expected {:?}", format!("d{}", "c".repeat(39)))),
            "{message}"
        );
        assert!(!message.contains(&"c".repeat(40)), "{message}");
    }

    #[test]
    fn a_text_mismatch_after_a_multi_byte_character_keeps_the_excerpt_valid() {
        let actual = format!("é{}", "x".repeat(50));
        let expected = format!("é{}", "y".repeat(50));
        let message = capture_panic(|| assert_same_text("the render", &actual, &expected));
        assert!(message.contains("first difference at byte 2"), "{message}");
        assert!(
            message.contains(&format!("actual {:?}", "x".repeat(40))),
            "{message}"
        );
        assert!(
            message.contains(&format!("expected {:?}", "y".repeat(40))),
            "{message}"
        );
        assert!(!message.contains(&"x".repeat(41)), "{message}");
    }

    #[test]
    fn a_text_window_that_ends_inside_a_multi_byte_character_stops_at_the_boundary() {
        // The window of the expected side ends inside the euro sign, so its
        // excerpt stops at the character boundary before the sign. The
        // window of the actual side ends on a boundary, so a window of 39 or
        // 41 bytes prints another excerpt and the message differs.
        let actual = format!("{}€{}xy{}", "a".repeat(39), "b".repeat(37), "tail");
        let expected = format!("{}☃{}€{}", "a".repeat(39), "b".repeat(36), "tail");
        let message = capture_panic(|| assert_same_text("the render", &actual, &expected));
        assert_eq!(
            message,
            format!(
                "the render: the texts differ (actual length {} bytes, expected length {} bytes); \
                 first difference at byte 40: actual {:?}, expected {:?}",
                actual.len(),
                expected.len(),
                &actual[39..79],
                &expected[39..78],
            )
        );
    }

    #[test]
    fn a_text_window_of_every_size_holds_a_different_excerpt() {
        // The euro sign of the expected side spans the bytes 76 to 79, so a
        // window of thirty-nine bytes ends inside the sign and stops at the
        // boundary at byte 76, a window of forty bytes ends at the boundary
        // at byte 79 and a window of forty-one bytes ends at the boundary
        // at byte 80. The actual side ends the window of thirty-nine bytes
        // at the boundary at byte 78, the window of forty bytes at the
        // boundary at byte 79 and the window of forty-one bytes at the
        // boundary at byte 80. The three windows therefore print three
        // different excerpts on either side, so a window that is one byte
        // shorter or one byte longer fails this test.
        let actual = format!("{}€{}xy{}", "a".repeat(39), "b".repeat(37), "tail");
        let expected = format!("{}☃{}€x{}", "a".repeat(39), "b".repeat(34), "tail");
        let message = capture_panic(|| assert_same_text("the render", &actual, &expected));
        assert_eq!(
            message,
            format!(
                "the render: the texts differ (actual length {} bytes, expected length {} bytes); \
                 first difference at byte 40: actual {:?}, expected {:?}",
                actual.len(),
                expected.len(),
                &actual[39..79],
                &expected[39..79],
            )
        );
    }

    #[test]
    fn the_text_excerpt_starts_at_the_difference_when_it_is_past_the_window() {
        let actual = format!("{}b{}", "a".repeat(100), "c".repeat(100));
        let expected = format!("{}d{}", "a".repeat(100), "c".repeat(100));
        let message = capture_panic(|| assert_same_text("the render", &actual, &expected));
        assert!(
            message.contains(&format!("actual {:?}", &actual[100..140])),
            "{message}"
        );
        assert!(
            message.contains(&format!("expected {:?}", &expected[100..140])),
            "{message}"
        );
        assert!(!message.contains(&"a".repeat(40)), "{message}");
    }

    #[test]
    fn equal_buffers_pass() {
        assert_same_bytes("the fixture", b"the same bytes", b"the same bytes");
    }

    #[test]
    fn a_buffer_mismatch_panics_with_the_first_difference() {
        let mut actual = vec![0x01_u8, 0xff];
        let mut expected = vec![0x02_u8, 0xfe];
        actual.extend_from_slice(&[b'x'; 41]);
        expected.extend_from_slice(&[b'x'; 41]);
        actual.push(0xfd);
        expected.push(0xfc);
        let message = capture_panic(|| assert_same_bytes("the fixture", &actual, &expected));
        assert_eq!(
            message,
            format!(
                "the fixture: the buffers differ (actual length {} bytes, expected length {} bytes); \
                 first difference at byte 0: actual {}, expected {}",
                actual.len(),
                expected.len(),
                escaped_bytes(&actual[0..40]),
                escaped_bytes(&expected[0..40]),
            )
        );
    }

    #[test]
    fn the_buffer_message_keeps_the_excerpts_short() {
        let mut actual = vec![b'a'; 100];
        let mut expected = vec![b'a'; 100];
        actual.push(0xff);
        expected.push(0xfe);
        actual.extend_from_slice(&[b'c'; 100]);
        expected.extend_from_slice(&[b'c'; 100]);
        let message = capture_panic(|| assert_same_bytes("the fixture", &actual, &expected));
        assert!(
            message.contains("first difference at byte 100"),
            "{message}"
        );
        assert!(
            message.contains(&format!("\\xff{}", "c".repeat(39))),
            "{message}"
        );
        assert!(!message.contains(&"c".repeat(40)), "{message}");
        assert!(!message.contains(&"a".repeat(40)), "{message}");
        assert_eq!(
            message,
            format!(
                "the fixture: the buffers differ (actual length {} bytes, expected length {} bytes); \
                 first difference at byte 100: actual {}, expected {}",
                actual.len(),
                expected.len(),
                escaped_bytes(&actual[100..140]),
                escaped_bytes(&expected[100..140]),
            )
        );
    }

    #[test]
    fn a_buffer_mismatch_inside_invalid_utf8_prints_escaped_bytes() {
        let mut actual = vec![b'a', 0xff];
        let mut expected = vec![b'a', 0xfe];
        actual.extend_from_slice(&[b'x'; 41]);
        actual.push(0xfd);
        expected.extend_from_slice(&[b'x'; 41]);
        expected.push(0xfc);
        let message = capture_panic(|| assert_same_bytes("the fixture", &actual, &expected));
        assert!(message.contains("first difference at byte 1"), "{message}");
        assert!(message.contains("\\xff"), "{message}");
        assert!(message.contains("\\xfe"), "{message}");
        assert!(!message.contains('\u{fffd}'), "{message}");
        assert_eq!(
            message,
            format!(
                "the fixture: the buffers differ (actual length {} bytes, expected length {} bytes); \
                 first difference at byte 1: actual {}, expected {}",
                actual.len(),
                expected.len(),
                escaped_bytes(&actual[1..41]),
                escaped_bytes(&expected[1..41]),
            )
        );
    }

    #[test]
    fn the_buffer_excerpt_holds_at_most_forty_bytes_of_each_side() {
        let mut actual = vec![b'a'; 100];
        actual.push(0xff);
        actual.push(0xfd);
        actual.extend_from_slice(&[b'y'; 40]);
        let mut expected = vec![b'a'; 100];
        expected.push(0xfe);
        expected.push(0xfc);
        expected.extend_from_slice(&[b'y'; 40]);
        let message = capture_panic(|| assert_same_bytes("the fixture", &actual, &expected));
        assert!(
            message.contains("first difference at byte 100"),
            "{message}"
        );
        assert!(message.contains("\\xff"), "{message}");
        assert!(!message.contains(&"a".repeat(40)), "{message}");
        assert_eq!(
            message,
            format!(
                "the fixture: the buffers differ (actual length {} bytes, expected length {} bytes); \
                 first difference at byte 100: actual {}, expected {}",
                actual.len(),
                expected.len(),
                escaped_bytes(&actual[100..140]),
                escaped_bytes(&expected[100..140]),
            )
        );
    }

    #[test]
    fn equal_json_values_pass() {
        assert_same_json("the body", &json!({"a": [1, 2]}), &json!({"a": [1, 2]}));
    }

    #[test]
    fn a_json_mismatch_panics_with_the_pointer() {
        let message =
            capture_panic(|| assert_same_json("the body", &json!({"a": 1}), &json!({"a": 2})));
        assert_eq!(
            message,
            "the body: the JSON values differ at \"/a\": actual 1, expected 2"
        );
    }

    #[test]
    fn a_json_kind_mismatch_prints_the_kind_and_the_length() {
        let message = capture_panic(|| {
            assert_same_json("the body", &json!({"a": [1, 2]}), &json!({"a": {"b": 1}}));
        });
        assert_eq!(
            message,
            "the body: the JSON values differ at \"/a\": actual an array of 2 items, \
             expected an object of 1 fields"
        );
    }

    #[test]
    fn a_json_root_mismatch_names_the_document_root() {
        let message =
            capture_panic(|| assert_same_json("the body", &json!([1, 2]), &json!({"a": 1})));
        assert_eq!(
            message,
            "the body: the JSON values differ at the document root: \
             actual an array of 2 items, expected an object of 1 fields"
        );
    }

    #[test]
    fn the_json_message_keeps_the_string_excerpts_short() {
        let actual_text = format!("{}b{}", "x".repeat(100), "y".repeat(100));
        let expected_text = format!("{}d{}", "x".repeat(100), "y".repeat(100));
        let actual = json!({"a": actual_text.clone()});
        let expected = json!({"a": expected_text.clone()});
        let message = capture_panic(|| assert_same_json("the body", &actual, &expected));
        assert!(message.contains("at \"/a\""), "{message}");
        assert!(message.contains("a string of 201 bytes"), "{message}");
        assert!(
            message.contains(&format!(
                "a string of {} bytes: {:?}",
                actual_text.len(),
                &actual_text[100..140]
            )),
            "{message}"
        );
        assert!(
            message.contains(&format!(
                "a string of {} bytes: {:?}",
                expected_text.len(),
                &expected_text[100..140]
            )),
            "{message}"
        );
        assert!(!message.contains(&"y".repeat(40)), "{message}");
    }

    #[test]
    fn a_json_string_that_differs_after_a_multi_byte_character_prints_the_excerpt() {
        let actual_text = format!("{}b{}", "é".repeat(20), "c".repeat(40));
        let expected_text = format!("{}d{}", "é".repeat(20), "c".repeat(40));
        let actual = json!({"a": actual_text.clone()});
        let expected = json!({"a": expected_text.clone()});
        let message = capture_panic(|| assert_same_json("the body", &actual, &expected));
        assert!(message.contains("at \"/a\""), "{message}");
        assert!(
            message.contains(&format!(
                "a string of {} bytes: {:?}",
                actual_text.len(),
                &actual_text[40..80]
            )),
            "{message}"
        );
        assert!(
            message.contains(&format!(
                "a string of {} bytes: {:?}",
                expected_text.len(),
                &expected_text[40..80]
            )),
            "{message}"
        );
        assert!(!message.contains(&"é".repeat(20)), "{message}");
    }

    #[test]
    fn the_json_excerpt_starts_at_the_difference_when_it_is_past_the_window() {
        let actual_text = format!("{}b{}", "x".repeat(100), "y".repeat(100));
        let expected_text = format!("{}d{}", "x".repeat(100), "y".repeat(100));
        let actual = json!({"a": actual_text.clone()});
        let expected = json!({"a": expected_text.clone()});
        let message = capture_panic(|| assert_same_json("the body", &actual, &expected));
        assert!(
            message.contains(&format!(
                "a string of {} bytes: {:?}",
                actual_text.len(),
                &actual_text[100..140]
            )),
            "{message}"
        );
        assert!(
            message.contains(&format!(
                "a string of {} bytes: {:?}",
                expected_text.len(),
                &expected_text[100..140]
            )),
            "{message}"
        );
        assert!(!message.contains(&"x".repeat(40)), "{message}");
    }

    #[test]
    fn a_json_pointer_of_the_bound_prints_whole() {
        let mut actual = serde_json::Map::new();
        actual.insert("k".repeat(79), json!(1));
        let mut expected = serde_json::Map::new();
        expected.insert("k".repeat(79), json!(2));
        let message = capture_panic(|| {
            assert_same_json("the body", &Value::Object(actual), &Value::Object(expected));
        });
        let pointer = format!("/{}", "k".repeat(79));
        assert_eq!(pointer.len(), 80);
        assert_eq!(
            message,
            format!("the body: the JSON values differ at {pointer:?}: actual 1, expected 2")
        );
    }

    #[test]
    fn a_json_pointer_one_byte_over_the_bound_prints_the_length() {
        let mut actual = serde_json::Map::new();
        actual.insert("k".repeat(80), json!(1));
        let mut expected = serde_json::Map::new();
        expected.insert("k".repeat(80), json!(2));
        let message = capture_panic(|| {
            assert_same_json("the body", &Value::Object(actual), &Value::Object(expected));
        });
        let pointer = format!("/{}", "k".repeat(80));
        assert_eq!(pointer.len(), 81);
        assert_eq!(
            message,
            format!(
                "the body: the JSON values differ at a pointer of {} bytes, starting with {:?}: \
                 actual 1, expected 2",
                pointer.len(),
                &pointer[..80],
            )
        );
    }

    #[test]
    fn a_json_key_longer_than_the_pointer_bound_prints_the_length() {
        let mut actual = serde_json::Map::new();
        actual.insert("k".repeat(100), json!(1));
        let mut expected = serde_json::Map::new();
        expected.insert("k".repeat(100), json!(2));
        let message = capture_panic(|| {
            assert_same_json("the body", &Value::Object(actual), &Value::Object(expected));
        });
        let pointer = format!("/{}", "k".repeat(100));
        assert_eq!(
            message,
            format!(
                "the body: the JSON values differ at a pointer of {} bytes, starting with {:?}: \
                 actual 1, expected 2",
                pointer.len(),
                &pointer[..80],
            )
        );
    }

    #[test]
    fn a_json_key_with_a_newline_prints_an_escaped_pointer() {
        let mut actual = serde_json::Map::new();
        actual.insert("a\nb".to_owned(), json!(1));
        let mut expected = serde_json::Map::new();
        expected.insert("a\nb".to_owned(), json!(2));
        let message = capture_panic(|| {
            assert_same_json("the body", &Value::Object(actual), &Value::Object(expected));
        });
        assert_eq!(
            message,
            "the body: the JSON values differ at \"/a\\nb\": actual 1, expected 2"
        );
        assert!(!message.contains('\n'), "{message}");
    }

    #[test]
    fn a_json_field_that_one_side_lacks_prints_missing() {
        let message =
            capture_panic(|| assert_same_json("the body", &json!({"a": null}), &json!({})));
        assert_eq!(
            message,
            "the body: the JSON values differ at \"/a\": actual null, expected missing"
        );
    }

    #[test]
    fn a_json_field_that_the_actual_side_lacks_prints_missing() {
        let message =
            capture_panic(|| assert_same_json("the body", &json!({}), &json!({"a": null})));
        assert_eq!(
            message,
            "the body: the JSON values differ at \"/a\": actual missing, expected null"
        );
    }

    #[test]
    fn equal_values_pass() {
        assert_same_values("the logits", &[1.0_f32, 2.0, 3.0], &[1.0_f32, 2.0, 3.0]);
    }

    #[test]
    fn equal_token_ids_pass() {
        assert_same_values("the token IDs", &[1_u32, 2, 3], &[1_u32, 2, 3]);
    }

    #[test]
    fn a_value_mismatch_names_the_index_and_the_two_values() {
        let message = capture_panic(|| {
            assert_same_values(
                "the logits",
                &[10.0_f32, 20.5, 30.0],
                &[10.0_f32, 20.0, 30.0],
            );
        });
        assert_eq!(
            message,
            "the logits: the values differ (actual length 3 values, expected length 3 values); \
             first difference at index 1: actual 20.5, expected 20.0"
        );
    }

    #[test]
    fn a_value_length_mismatch_names_both_lengths() {
        let message = capture_panic(|| {
            assert_same_values("the logits", &[1.0_f32, 2.0], &[1.0_f32, 2.0, 3.0]);
        });
        assert_eq!(
            message,
            "the logits: the values differ (actual length 2 values, expected length 3 values); \
             first difference at index 2: actual missing, expected 3.0"
        );
    }

    #[test]
    fn a_value_mismatch_holds_no_other_value() {
        let message = capture_panic(|| {
            assert_same_values(
                "the logits",
                &[10.0_f32, 20.5, 30.0],
                &[10.0_f32, 20.0, 30.0],
            );
        });
        assert!(!message.contains("10"), "{message}");
        assert!(!message.contains("30"), "{message}");
    }
}
