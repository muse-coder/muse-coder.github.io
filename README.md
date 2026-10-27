# Di Mou's Homepage

Source code for [muse-coder.github.io](https://muse-coder.github.io/), built with Jekyll and the [Minimal Light](https://github.com/yaoyao-liu/minimal-light) theme.

## Bilingual blog

The Blog lives at `/blog/`. Posts are stored in `_posts/` as paired English and
Chinese Markdown files. Each pair shares a `translation_key` and uses `lang: en`
or `lang: zh-CN`; the layouts generate language links and `hreflang` metadata.
Use a stable `permalink` for each translation and place shared images under
`assets/img/blog/`. The index lists each English post once, alongside its Chinese
translation.
