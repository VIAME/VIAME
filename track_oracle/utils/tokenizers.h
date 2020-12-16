// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_TOKENIZERS_H
#define INCL_TOKENIZERS_H

#include <vital/vital_config.h>
#include <track_oracle/utils/track_oracle_tokenizers_export.h>

#include <iosfwd>
#include <string>
#include <vector>

namespace kwiver {
namespace track_oracle {

namespace csv_tokenizer
{
  TRACK_ORACLE_TOKENIZERS_EXPORT std::istream& get_record(std::istream& is, std::vector<std::string>& out);
  TRACK_ORACLE_TOKENIZERS_EXPORT std::vector<std::string> parse(std::string const& s);
};

namespace xml_tokenizer
{
  TRACK_ORACLE_TOKENIZERS_EXPORT std::vector< std::string > first_n_tokens( const std::string& fn, size_t n = 1 );
};

} // ...track_oracle
} // ...kwiver

#endif
