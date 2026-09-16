// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "category_hierarchy.h"

// Upstream reaches rapidjson through cereal's vendored copy. P8-T06
// removed cereal and made `file_io/json.h` the one place rapidjson is
// included, so that the three behaviour flags cereal used to set on the
// way past are stated rather than inherited. Same headers, one door.
#include <viame/file_io/json.h>

#include <cctype>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace viame {

namespace {

// TXT keeps the historic whitespace-separated synonym syntax. CSV uses one
// category per row, followed by synonyms or :parent= fields.
std::vector< std::vector< std::string > >
read_label_rows( std::istream& in, bool csv )
{
  std::vector< std::vector< std::string > > rows;
  std::vector< std::string > row;
  std::string field;
  char quote = 0;
  bool started = false;
  bool closed = false;
  bool quoted = false;
  auto finish_field = [&]() {
    if( csv && !quoted )
    {
      const auto last = field.find_last_not_of( " \t" );
      field.erase( last == std::string::npos ? 0 : last + 1 );
    }
    if( started )
    {
      if( field.empty() )
      {
        throw std::runtime_error( "Empty category name or synonym" );
      }
      row.push_back( field );
    }
    field.clear();
    started = closed = quoted = false;
  };
  auto finish_row = [&]() {
    finish_field();
    if( !row.empty() )
    {
      rows.push_back( row );
      row.clear();
    }
  };
  char c;
  while( in.get( c ) )
  {
    if( quote )
    {
      if( c == quote )
      {
        if( in.peek() == quote )
        {
          in.get();
          field += c;
        }
        else
        {
          quote = 0;
          closed = true;
        }
      }
      else if( !csv && c == '\\' &&
               ( in.peek() == quote || in.peek() == '\\' ) )
      {
        field += static_cast< char >( in.get() );
      }
      else
      {
        if( !csv && ( c == '\n' || c == '\r' ) )
        {
          throw std::runtime_error( "Unterminated quoted label" );
        }
        field += c;
      }
    }
    else if( c == '\n' || c == '\r' )
    {
      if( c == '\r' && in.peek() == '\n' ) { in.get(); }
      finish_row();
    }
    else if( !csv && c == '#' )
    {
      while( in.peek() != '\n' && in.peek() != '\r' && in.peek() != EOF )
      {
        in.get();
      }
      finish_row();
    }
    else if( csv && c == ',' )
    {
      if( !started && row.empty() )
      {
        throw std::runtime_error( "Empty category name" );
      }
      finish_field();
    }
    else if( std::isspace( static_cast< unsigned char >( c ) ) )
    {
      if( !csv ) { finish_field(); }
      else if( started && !closed ) { field += c; }
    }
    else if( ( c == '"' || ( !csv && c == '\'' ) ) &&
             ( !started || ( !csv && field == ":parent=" ) ) )
    {
      quote = c;
      started = quoted = true;
    }
    else
    {
      if( closed )
      {
        throw std::runtime_error( "Expected a separator after quoted label" );
      }
      field += c;
      started = true;
    }
  }
  if( quote ) { throw std::runtime_error( "Unterminated quoted label" ); }
  finish_row();
  return rows;
}

std::string json_label( const rapidjson::Value& value )
{
  if( !value.IsString() || value.GetStringLength() == 0 )
  {
    throw std::runtime_error( "Category names must be nonempty strings" );
  }
  return std::string( value.GetString(), value.GetStringLength() );
}

} // namespace

// ----------------------------------------------------------------------------
category_hierarchy
::category_hierarchy()
{}

category_hierarchy
::category_hierarchy( std::string filename )
{
  this->load_from_file( filename );
}

category_hierarchy
::category_hierarchy(
  const label_vec_t& class_names,
  const label_vec_t& parent_names,
  const label_id_vec_t& ids )
{
  if( !parent_names.empty() && class_names.size() != parent_names.size() )
  {
    throw std::invalid_argument( "Parameter vector sizes differ." );
  }

  if( !ids.empty() && class_names.size() != ids.size() )
  {
    throw std::invalid_argument( "Parameter vector sizes differ." );
  }

  if( class_names.empty() )
  {
    throw std::invalid_argument( "Parameter vector are empty." );
  }

  for( size_t i = 0; i < class_names.size(); ++i )
  {
    const label_t& name = class_names[ i ];

    this->add_class( name );

    if( !ids.empty() )
    {
      m_hierarchy[ name ]->category_id = ids[ i ];
    }
  }

  if( !parent_names.empty() )
  {
    for( size_t i = 0; i < class_names.size(); ++i )
    {
      if( !parent_names[ i ].empty() )
      {
        this->add_relationship( class_names[ i ], parent_names[ i ] );
      }
    }
  }
}

category_hierarchy
::~category_hierarchy()
{}

// ----------------------------------------------------------------------------
void
category_hierarchy
::add_class(
  const label_t& class_name,
  const label_t& parent_name,
  const label_id_t id )
{
  if( m_hierarchy.find( class_name ) != m_hierarchy.end() )
  {
    throw std::runtime_error( "Category already exists." );
  }

  category_sptr new_entry( new category() );
  m_hierarchy[ class_name ] = new_entry;

  new_entry->category_name = class_name;
  new_entry->category_id = id;

  if( !parent_name.empty() )
  {
    hierarchy_const_itr_t itr = find( parent_name );
    new_entry->parents.push_back( itr->second.get() );
  }
}

// ----------------------------------------------------------------------------
bool
category_hierarchy
::has_class_name( const std::string& class_name ) const
{
  if( m_hierarchy.find( class_name ) != m_hierarchy.end() )
  {
    return true;
  }
  return false;
}

// ----------------------------------------------------------------------------
category_hierarchy::label_t
category_hierarchy
::get_class_name( const label_t& class_name ) const
{
  hierarchy_const_itr_t itr = this->find( class_name );

  return itr->second->category_name;
}

// ----------------------------------------------------------------------------
category_hierarchy::label_vec_t
category_hierarchy
::get_class_synonyms( const label_t& class_name ) const
{
  return this->find( class_name )->second->synonyms;
}

// ----------------------------------------------------------------------------
category_hierarchy::label_id_t
category_hierarchy
::get_class_id( const label_t& class_name ) const
{
  hierarchy_const_itr_t itr = this->find( class_name );

  return itr->second->category_id;
}

// ----------------------------------------------------------------------------
category_hierarchy::label_vec_t
category_hierarchy
::get_class_parents( const label_t& class_name ) const
{
  label_vec_t output;

  hierarchy_const_itr_t itr = this->find( class_name );

  for( category* p : itr->second->parents )
  {
    output.push_back( p->category_name );
  }

  return output;
}

// ----------------------------------------------------------------------------
void
category_hierarchy
::add_relationship( const label_t& child_name, const label_t& parent_name )
{
  hierarchy_const_itr_t itr1 = this->find( child_name );
  hierarchy_const_itr_t itr2 = this->find( parent_name );

  itr1->second->parents.push_back( itr2->second.get() );
  itr2->second->children.push_back( itr1->second.get() );
}

// ----------------------------------------------------------------------------
void
category_hierarchy
::add_synonym( const label_t& class_name, const label_t& synonym_name )
{
  hierarchy_const_itr_t itr = this->find( class_name );

  if( has_class_name( synonym_name ) )
  {
    throw std::runtime_error( "Synonym name already exists in hierarchy" );
  }

  itr->second->synonyms.push_back( synonym_name );
  m_hierarchy[ synonym_name ] = itr->second;
}

// ----------------------------------------------------------------------------
category_hierarchy::label_vec_t
category_hierarchy
::all_class_names() const
{
  std::vector< category_sptr > sorted_cats = sorted_categories();

  label_vec_t names;

  for( category_sptr c : sorted_cats )
  {
    names.push_back( c->category_name );
  }

  return names;
}

// ----------------------------------------------------------------------------
category_hierarchy::label_vec_t
category_hierarchy
::child_class_names() const
{
  std::vector< category_sptr > sorted_cats = sorted_categories();

  label_vec_t names;

  for( category_sptr c : sorted_cats )
  {
    if( c->children.empty() )
    {
      names.push_back( c->category_name );
    }
  }

  return names;
}

// ----------------------------------------------------------------------------
size_t
category_hierarchy
::size() const
{
  return m_hierarchy.size();
}

// ----------------------------------------------------------------------------
void
category_hierarchy
::load_from_file( const std::string& filename )
{
  std::ifstream in( filename.c_str(), std::ios::binary );

  if( !in )
  {
    throw std::runtime_error( "Unable to open " + filename );
  }

  // Strip a UTF-8 BOM before reading either text or JSON.
  if( in.peek() == 0xef )
  {
    char bom[3] = {};
    in.read( bom, 3 );
    if( std::string( bom, 3 ) != "\xef\xbb\xbf" )
    {
      throw std::runtime_error( "Invalid UTF-8 BOM in " + filename );
    }
  }
  std::string extension;
  const auto dot = filename.rfind( '.' );
  if( dot != std::string::npos ) { extension = filename.substr( dot ); }
  std::transform( extension.begin(), extension.end(), extension.begin(),
    []( unsigned char c ) { return std::tolower( c ); } );

  std::vector< std::pair< label_t, label_t > > relationships;
  int entry_num = 0;
  if( extension == ".json" )
  {
    const std::string text( ( std::istreambuf_iterator< char >( in ) ),
                            std::istreambuf_iterator< char >() );
    rapidjson::Document doc;
    doc.Parse( text.data(), text.size() );
    if( doc.HasParseError() )
    {
      throw std::runtime_error( "Invalid label JSON in " + filename + ": " +
        rapidjson::GetParseError_En( doc.GetParseError() ) );
    }
    const rapidjson::Value empty_categories( rapidjson::kArrayType );
    const rapidjson::Value& categories =
      doc.IsObject() && doc.HasMember( "categories" ) ? doc["categories"] :
      doc.IsObject() && doc.HasMember( "typeHierarchy" ) ? empty_categories : doc;
    if( !categories.IsArray() )
    {
      throw std::runtime_error( "Label JSON must be an array or contain a categories array" );
    }
    for( const auto& category : categories.GetArray() )
    {
      const auto name = json_label( category.IsObject() && category.HasMember( "name" )
        ? category["name"] : category );
      int id = entry_num++;
      if( category.IsObject() && category.HasMember( "id" ) )
      {
        if( !category["id"].IsInt() )
        {
          throw std::runtime_error( "Category id must be an integer" );
        }
        id = category["id"].GetInt();
      }
      this->add_class( name, "", id );
      // Match DIVE's COCO convention: a nonempty supercategory takes
      // precedence over parents. Synonyms remain aliases, never parents.
      bool has_supercategory = false;
      if( category.IsObject() && category.HasMember( "supercategory" ) )
      {
        const auto& parent = category["supercategory"];
        if( !parent.IsString() )
        {
          throw std::runtime_error( "supercategory must be a string" );
        }
        if( parent.GetStringLength() )
        {
          relationships.emplace_back( name, json_label( parent ) );
          has_supercategory = true;
        }
      }
      for( const std::string key : { "synonyms", "parents" } )
      {
        if( ( key == "parents" && has_supercategory ) ||
            !category.IsObject() || !category.HasMember( key.c_str() ) ) { continue; }
        const auto& values = category[key.c_str()];
        if( !values.IsArray() )
        {
          throw std::runtime_error( key + " must be an array of strings" );
        }
        for( const auto& value : values.GetArray() )
        {
          if( key == "synonyms" ) { this->add_synonym( name, json_label( value ) ); }
          else { relationships.emplace_back( name, json_label( value ) ); }
        }
      }
    }
    if( doc.IsObject() && doc.HasMember( "typeHierarchy" ) &&
        !doc["typeHierarchy"].IsNull() )
    {
      const auto& hierarchy = doc["typeHierarchy"];
      if( !hierarchy.IsObject() )
      {
        throw std::runtime_error( "typeHierarchy must be a child-to-parent object" );
      }
      for( auto edge = hierarchy.MemberBegin(); edge != hierarchy.MemberEnd(); ++edge )
      {
        relationships.emplace_back( json_label( edge->name ), json_label( edge->value ) );
      }
    }
    // DIVE permits hierarchy-only nodes (COCO supercategories often have no
    // category record). Preserve these nodes after explicitly listed classes.
    for( const auto& edge : relationships )
    {
      for( const auto& name : { edge.first, edge.second } )
      {
        if( !this->has_class_name( name ) ) { this->add_class( name ); }
      }
    }
  }
  else
  {
    for( const auto& tokens : read_label_rows( in, extension == ".csv" ) )
    {
      this->add_class( tokens[0], "", entry_num++ );
      for( size_t i = 1; i < tokens.size(); ++i )
      {
        if( tokens[i].compare( 0, 8, ":parent=" ) == 0 )
        {
          relationships.emplace_back( tokens[0], tokens[i].substr( 8 ) );
        }
        else { this->add_synonym( tokens[0], tokens[i] ); }
      }
    }
  }

  for( auto rel : relationships )
  {
    const auto child = this->get_class_name( rel.first );
    const auto parent = this->get_class_name( rel.second );
    std::vector< label_t > pending{ parent };
    std::set< label_t > visited;
    while( !pending.empty() )
    {
      const auto node = pending.back();
      pending.pop_back();
      if( node == child )
      {
        throw std::runtime_error( "Cycle in category hierarchy: " + child + " -> " + parent );
      }
      if( visited.insert( node ).second )
      {
        const auto parents = this->get_class_parents( node );
        pending.insert( pending.end(), parents.begin(), parents.end() );
      }
    }
    const auto parents = this->get_class_parents( child );
    if( std::find( parents.begin(), parents.end(), parent ) == parents.end() )
    {
      this->add_relationship( child, parent );
    }
  }
}

// ----------------------------------------------------------------------------
category_hierarchy::hierarchy_const_itr_t
category_hierarchy
::find( const label_t& lbl ) const
{
  hierarchy_const_itr_t itr = m_hierarchy.find( lbl );

  if( itr == m_hierarchy.end() )
  {
    throw std::runtime_error( "Class node " + lbl + " does not exist." );
  }

  return itr;
}

// ----------------------------------------------------------------------------
std::vector< category_hierarchy::category_sptr >
category_hierarchy
::sorted_categories() const
{
  std::vector< category_sptr > sorted_cats;

  for( hierarchy_const_itr_t p = m_hierarchy.begin();
       p != m_hierarchy.end(); ++p )
  {
    if( p->first == p->second->category_name ) // don't include synonyms
    {
      sorted_cats.push_back( p->second );
    }
  }

  std::sort(
    sorted_cats.begin(), sorted_cats.end(),
    []( const category_sptr& lhs, const category_sptr& rhs ){
      return ( lhs->category_id >= 0 && rhs->category_id >= 0 &&
               lhs->category_id < rhs->category_id ) ||
             ( lhs->category_id >= 0 && rhs->category_id < 0 ) ||
             ( lhs->category_id < 0 && rhs->category_id < 0 &&
               lhs->category_name < rhs->category_name );
    } );

  return sorted_cats;
}

} // namespace viame
