int main()
{
  static constexpr char convert[] = "0123456789abcdef";

  if ( '3' != convert[3] ) return 1;

  return 0;
}
