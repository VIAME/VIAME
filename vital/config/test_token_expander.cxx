/*ckwg +5
 * Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <token_expander.h>
#include <token_type_env.h>
#include <token_type_sysenv.h>
#include <token_type_symtab.h>
#include <token_stream_filter.h>
#include <sstream>


int main(int argc, char *argv[])
{

  vidtk::token_expander exp;

  exp.add_token_type( new vidtk::ns_token_expand::token_type_env() );
  exp.add_token_type( new vidtk::ns_token_expand::token_type_sysenv() );


  std::string buffer("asdfasdf $ENV{HOME} ASDFASDF");
  std::cout << buffer << " =>> " << exp.expand_token( buffer ) << std::endl;

  buffer = "asdfasdfasdf$ENV{xxx}ASDFASDF";
  std::cout << buffer << " =>> " << exp.expand_token( buffer ) << std::endl;

  buffer = "asdfasdfasdf$ENVST{xxx}ASDFASDF";
  std::cout << buffer << " =>> " << exp.expand_token( buffer ) << std::endl;

  buffer = "asdfasdfasdf$ENV{HOME} favored editor $ENV{EDITOR}";
  std::cout << buffer << " =>> " << exp.expand_token( buffer ) << std::endl;

  buffer = "$ENV{HOME} favored editor $ENV{EDITOR}";
  std::cout << buffer << " =>> " << exp.expand_token( buffer ) << std::endl;

  buffer = "$ENV{HOME}";
  std::cout << buffer << " =>> " << exp.expand_token( buffer ) << std::endl;

  buffer = "system has $SYSENV{numproc} processors";
  std::cout << buffer << " =>> " << exp.expand_token( buffer ) << std::endl;


  // test stream filter
  boost::iostreams::filtering_istream in_stream;
  std::stringstream data_stream;

  buffer = "asdfasdf $ENV{HOME} ASDFASDF\n"
    "asdfasdfasdf$ENV{xxx}ASDFASDF\n"
    "asdfa--s--d==fasdf$ENVST{xxx}ASDFASDF\n"
    "asdfasdfasdf$ENV{HOME} favored editor $ENV{EDITOR}\n"
    "system has $SYSENV{numproc} processors\n";

  data_stream.str(buffer);
  std::cout << "input buffer: " << buffer;
  std::cout << buffer << " =>> " << exp.expand_token( buffer ) << std::endl;

  std::cout << "--- testing expander stream ---\n";

  // add filter with token expander
  vidtk::token_expander* exp_ptr( new vidtk::token_expander);
  exp_ptr->add_token_type( new vidtk::ns_token_expand::token_type_env() );
  exp_ptr->add_token_type( new vidtk::ns_token_expand::token_type_sysenv() );

  // need to test config reader. TBD


  in_stream.push (vidtk::token_stream_filter( exp_ptr ));
  // in_stream.push (vidtk::shell_comments_filter());
  // in_stream.push (vidtk::blank_line_filter());
  in_stream.push( data_stream );

  // read file until eof
  while (! in_stream.eof() )
  {
    std::string line;
    std::getline (in_stream, line);
    // "line" has tokens expanded

    if ( in_stream.eof() ) break;

    // process 'line'
    std::cout << "line from filter: " << line << "\n";
  } // end while

  return 0;
}
