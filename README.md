# DHARD - Data structure for writing shard that persists on disk

Dhard provides data structures and traits for writing shard that persists on disk.
Unlike most of libraries, the business logic/implementation is up to you.
Why? Dhard is made to work with **any** type (note that to be able to read
on disk it the type should be sized).
Its main data structure is `Shard<T, V>` where T is
a _marker, a PhantomData which you can visualize as
the shard name

```rust
/// This is a sample type
#[derive(Debug, Default, Clone, PartialEq)]
struct TestData {
    id: u32,
    name: String,
}

struct Test;

/// Here Test is used as the Shard marker, it "names" the shard implementation/type
type TestShard = Shard<Test, TestData>;
```

To learn more about the library, you can check the [docs](https://docs.rs/dhard/0.1.0/dhard)

## LICENSE

Dhard is licensed as MIT or APACHE 2.0, you will find the licenses
at:

- [MIT LICENSE](./LICENSE-MIT)
- [APACHE 2.0 LICENSE](./LICENSE-APACHE)
