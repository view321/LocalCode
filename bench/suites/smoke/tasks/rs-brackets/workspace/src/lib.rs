//! Bracket balance checking for `()`, `[]`, `{}`.

/// True when every bracket in `s` opens and closes in the right order and
/// kind — including that nothing is left open at the end of the input.
/// Characters other than the six bracket characters are ignored.
pub fn balanced(s: &str) -> bool {
    let mut stack: Vec<char> = Vec::new();
    for c in s.chars() {
        match c {
            '(' | '[' | '{' => stack.push(c),
            ')' | ']' | '}' => {
                let Some(open) = stack.pop() else {
                    return false;
                };
                let expected = match c {
                    ')' => '(',
                    ']' => '[',
                    _ => '{',
                };
                if open != expected {
                    return false;
                }
            }
            _ => {}
        }
    }
    true
}
