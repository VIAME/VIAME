struct Foo
{
  Foo() {}
  Foo( const Foo& ) = default;
  virtual ~Foo() = default; // c++11 feature
};

struct Bar
{
  Bar() {}
  Bar( const Bar& ) = delete;
  Bar& operator=( const Bar& ) = delete;
  virtual ~Bar() {}
};

int
main()
{
  Foo foo;
  Bar bar;

  return 0;
}
