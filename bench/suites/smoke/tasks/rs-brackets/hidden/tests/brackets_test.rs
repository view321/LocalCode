use brackets::balanced;

#[test]
fn accepts_balanced_inputs() {
    assert!(balanced(""));
    assert!(balanced("()"));
    assert!(balanced("([]{})"));
    assert!(balanced("fn main() { let v = vec![1, (2)]; }"));
    assert!(balanced("({[]})"));
}

#[test]
fn rejects_wrong_kind_or_order() {
    assert!(!balanced("(]"));
    assert!(!balanced(")("));
    assert!(!balanced("([)]"));
    assert!(!balanced("}"));
}

#[test]
fn rejects_unclosed_openers() {
    assert!(!balanced("("));
    assert!(!balanced("(["));
    assert!(!balanced("((("));
    assert!(!balanced("text { unclosed"));
    assert!(!balanced("([]"));
}

#[test]
fn ignores_other_characters() {
    assert!(balanced("no brackets at all"));
    assert!(balanced("a(b)c[d]e{f}g"));
    assert!(!balanced("a(b]c"));
}
