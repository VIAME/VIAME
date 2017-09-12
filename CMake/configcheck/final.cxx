class foo final
{

};

struct Base
{
  virtual ~Base();
  virtual void foo();
};

struct A : Base
{
  virtual void foo() final; // A::foo is final
//   void bar() final; // Error: non-virtual function cannot be final
};

struct B final : A // struct B is final
{
//   void foo(); // Error: foo cannot be overridden as it's final in A
};

int main()
{
  return 0;
}
