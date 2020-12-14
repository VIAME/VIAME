// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "tokenizers.h"

#include <istream>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::istream;
using std::string;
using std::stringstream;
using std::vector;
using std::fread;

namespace // anon
{

struct xml_stream_buffer
{
  xml_stream_buffer( FILE* fptr );
  ~xml_stream_buffer();
  bool get_next_char( char& c );

private:
  FILE* fp;
  size_t index, top, bufsize;
  char* buf;
};

xml_stream_buffer
::xml_stream_buffer( FILE* fptr )
  : fp( fptr ), index(0), top(0), bufsize( 2048 )
{
  buf = new char[bufsize];
}

xml_stream_buffer
::~xml_stream_buffer()
{
  delete [] buf;
}

bool
xml_stream_buffer
::get_next_char( char& c )
{
  // if we're at the end of the buffer, try to top it up
  if (index == top)
  {
    // if fp is null, we're done
    if (fp == NULL) return false;

    index = 0;
    top = fread( buf, sizeof(char), bufsize, fp );
    // if we couldn't get a full buffer, then it's either error or EOF
    // either way, we'll be done as soon as this buffer is empty
    if ( top < bufsize ) fp = NULL;

    // if we're STILL at the end of the buffer, we're done
    if (index == top) return false;
  }

  c = buf[ index++ ];
  return true;
}

} // ...anon

namespace kwiver {
namespace track_oracle {

namespace xml_tokenizer {

vector< string>
first_n_tokens( const string& fn, size_t n )
{
  FILE* fp = fopen( fn.c_str(), "rb" );
  if (fp == NULL)
  {
    return vector<string>();
  }

  xml_stream_buffer xmlbuf( fp );

  vector< string > tokens;
  string this_token = "";
  // state machine:
  // 0 = init (nothing seen)
  // 1 = in a run of whitespace
  // 2 = in a run of not-whitespace
  // 0->1 or 1->2 transitions signal start of a token
  // 2->1 transitions signal completion of a token
  unsigned state = 0;
  bool in_comment = false;
  do
  {
    char this_char;
    if (! xmlbuf.get_next_char( this_char ))
    {
      // EOF / error / whatever; if outside comments, push current token (if any) and exit
      if (( ! in_comment) && (this_token != "")) tokens.push_back( this_token );
      break;
    }

    if (isspace( this_char )) // new state will be 1
    {
      switch (state)
      {
      case 0:
        state = 1;
        break;
      case 1:
        break;
      case 2:
        // check for entering / leaving comments; only store tokens which are
        // outside comments
        if (! in_comment)
        {
          if (this_token == "<!--")
          {
            in_comment = true;
          }
          else
          {
            tokens.push_back( this_token );
          }
        }
        else
        {
          if ((this_token.size() >= 3)
              && (this_token.substr( this_token.size()-3, 3) == "-->"))
          {
            in_comment = false;
          }
        }
        this_token = "";
        state = 1;
        break;
      }
    }
    else  // new state will be 2
    {
      state = 2;
      // quick-n-dirty approach to breaking on "raw" xml streams
      // which may be '<token1><token2><token3>':

      if ((this_char == '<') && (! in_comment))
      {
        tokens.push_back( this_token );
        this_token = "<";
      }
      else if (this_char == '>')
      {
        // does this end a comment?
        this_token += this_char;
        bool this_ends_comment =
          in_comment &&
          (this_token.size() >= 3) &&
          (this_token.substr( this_token.size()-3, 3) == "-->");
        if (in_comment && this_ends_comment )
        {
          // clear the token, but do not store
          this_token = "";
          in_comment = false;
        }
        else
        {
          // store && clear
          tokens.push_back( this_token );
          this_token = "";
        }
      }
      else
      {
        this_token += this_char;
        if (this_token == "<!--")
        {
          in_comment = true;
        }
      }
    }

    // don't store empty tokens
    if ( ! tokens.empty() )
    {
      if (tokens.back() == "")
      {
        tokens.pop_back();
      }
    }
  }
  while (tokens.size() < n );
  fclose( fp );
  return tokens;

}

} //...xml_tokenizer

namespace csv_tokenizer {

istream&
get_record(istream& is, vector<string>& values)
{
  values.clear();

  string token;
  char c;
  bool in_quote = false;

  is.get(c);
  if (is.eof()) return is; // Handle end of file gracefully

  for (; is; is.get(c))
  {
    // Handle DOS and (old-style) Mac line endings
    if (c == '\r')
    {
      if (is.peek() == '\n') continue; // CR before LF is ignored
      c = '\n'; // CR followed by anything else is treated as LF
    }
    else if (c == '"')
    {
      // Value starting with quote character denotes a quoted value
      if (token.empty())
      {
        in_quote = true;
        continue;
      }
      // Otherwise, quotes have special meaning only in quoted value
      else if (in_quote)
      {
        if (is.get(c))
        {
          // Doubled quote is emitted as-is; anything else ends quoting
          in_quote = (c == '"');
        }
        else
        {
          // Stream is now at EOF or in bad state; former is treated as end of
          // record; latter is left to caller to check and report an error
          break;
        }
      }
    }

    if (!in_quote)
    {
      if (c == '\n') break; // End of record
      if (c == ';' || c == ',')
      {
        values.push_back(token);
        token.clear();
        continue;
      }
    }

    token.push_back(c);
  }

  values.push_back(token);
  return is;
}

vector<string>
parse(string const& s)
{
  vector<string> result;

  stringstream ss(s);
  if (!get_record(ss, result) && !ss.eof())
    LOG_WARN( main_logger, "CSV parsing failed for input string '" << s << "'");

  return result;
}

} // ...csv_tokenizer

} // ...track_oracle
} // ...kwiver
