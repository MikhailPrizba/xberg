#![cfg(feature = "html")]

use xberg::ConversionOptions;
use xberg::extraction::convert_html_to_markdown;

#[test]
fn should_keep_accessible_names_without_adding_a_public_options_field() {
    let html = r#"<p>Before <a href="https://example.com" aria-label="Accessible name"></a> after.</p>
        <svg aria-label="Graphic label"><title>Graphic title</title><text>Drawn words</text></svg>
        <img src="photo.png" alt="Image description">"#;
    let output = convert_html_to_markdown(html, Some(ConversionOptions::default()), None).unwrap();
    for expected in ["Accessible name", "Graphic label", "Drawn words", "Image description"] {
        assert_eq!(output.matches(expected).count(), 1, "{expected}: {output}");
    }
}
