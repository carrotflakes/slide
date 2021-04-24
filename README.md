# SLIDE in Rust

This is an implementation of the [SLIDE (Sub-LInear Deep learning Engine)](https://arxiv.org/abs/1903.03129), ported from https://github.com/keroro824/HashingDeepLearning :kissing_heart:.

## Test with Amazon-670K dataset

Requirements:

- Rust 1.51.0+
- Machine: 10GB RAM
- [Amazon-670K dataset](https://github.com/keroro824/HashingDeepLearning)

```
$ git clone https://github.com/carrotflakes/slide.git
$ cd slide
$ cargo run --release
```

## License

Licensed under the MIT License.
