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
  virtual ~Bar() = delete;
};

int main()
{
  Foo foo;
  Bar bar;

  return 0;
}
