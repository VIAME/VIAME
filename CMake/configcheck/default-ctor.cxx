struct Foo
{
  Foo() = default;
  Foo(const Foo&) = default;
  ~Foo() = default;
};

struct Bar
{
  Bar() = delete;
  Bar(const Bar&) = delete;
};

int main()
{
  return 0;
}
