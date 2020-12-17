// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "category_hierarchy.h"

#include <vital/util/data_stream_reader.h>

#include <string>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iterator>
#include <utility>

namespace kwiver {
namespace vital {

// -----------------------------------------------------------------------------
category_hierarchy
::category_hierarchy()
{
}

category_hierarchy
::category_hierarchy( std::string filename )
{
  this->load_from_file( filename );
}

category_hierarchy
::category_hierarchy( const label_vec_t& class_names,
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

  for( unsigned i = 0; i < class_names.size(); ++i )
  {
    const label_t& name = class_names[i];

    this->add_class( name );

    if( !ids.empty() )
    {
      m_hierarchy[ name ]->category_id = ids[i];
    }
  }

  if( !parent_names.empty() )
  {
    for( unsigned i = 0; i < class_names.size(); ++i )
    {
      if( !parent_names[i].empty() )
      {
        this->add_relationship( class_names[i], parent_names[i] );
      }
    }
  }
}

category_hierarchy
::~category_hierarchy()
{
}

// -----------------------------------------------------------------------------
void
category_hierarchy
::add_class( const label_t& class_name,
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

// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
category_hierarchy::label_t
category_hierarchy
::get_class_name( const label_t& class_name ) const
{
  hierarchy_const_itr_t itr = this->find( class_name );

  return itr->second->category_name;
}

// -----------------------------------------------------------------------------
category_hierarchy::label_id_t
category_hierarchy
::get_class_id( const label_t& class_name ) const
{
  hierarchy_const_itr_t itr = this->find( class_name );

  return itr->second->category_id;
}

// -----------------------------------------------------------------------------
category_hierarchy::label_vec_t
category_hierarchy
::get_class_parents( const label_t& class_name ) const
{
  label_vec_t output;

  hierarchy_const_itr_t itr = this->find( class_name );

  for( category *p : itr->second->parents )
  {
    output.push_back( p->category_name );
  }

  return output;
}

// -----------------------------------------------------------------------------
void
category_hierarchy
::add_relationship( const label_t& child_name, const label_t& parent_name )
{
  hierarchy_const_itr_t itr1 = this->find( child_name );
  hierarchy_const_itr_t itr2 = this->find( parent_name );

  itr1->second->parents.push_back( itr2->second.get() );
  itr2->second->children.push_back( itr1->second.get() );
}

// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
size_t
category_hierarchy
::size() const
{
  return m_hierarchy.size();
}

// -----------------------------------------------------------------------------
void
category_hierarchy
::load_from_file( const std::string& filename )
{
  std::ifstream in( filename.c_str() );

  if( !in )
  {
    throw std::runtime_error( "Unable to open " + filename );
  }

  std::vector< std::pair< label_t, label_t > > relationships;

  std::string line;
  label_t label;

  int entry_num = 0;

  kwiver::vital::data_stream_reader dsr( in );
  while( dsr.getline( line ) )
  {
    std::vector< label_t > tokens;
    std::istringstream iss( line );
    std::copy( std::istream_iterator< std::string >( iss ),
      std::istream_iterator< std::string >(),
      std::back_inserter( tokens ) );

    if( tokens.size() == 0 || tokens[0].size() == 0 || tokens[0][0] == '#' )
    {
      continue;
    }

    this->add_class( tokens[0], "", entry_num );
    entry_num++;

    for( unsigned i = 1; i < tokens.size(); ++i )
    {
      if( tokens[i].compare( 0, 8, ":parent=" ) == 0 )
      {
        relationships.push_back(
          std::make_pair< label_t, label_t >(
            label_t( tokens[0] ), label_t( tokens[i].substr( 8 ) ) ) );
      }
      else
      {
        this->add_synonym( tokens[0], tokens[i] );
      }
    }
  }

  for( auto rel : relationships )
  {
    this->add_relationship( rel.first, rel.second );
  }
}

// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
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

  std::sort( sorted_cats.begin(), sorted_cats.end(),
    []( const category_sptr& lhs, const category_sptr& rhs )
      { return ( lhs->category_id >= 0 && rhs->category_id >= 0
                  && lhs->category_id < rhs->category_id ) ||
               ( lhs->category_id >= 0 && rhs->category_id < 0 ) ||
               ( lhs->category_id < 0 && rhs->category_id < 0
                  && lhs->category_name < rhs->category_name ); } );

  return sorted_cats;
}

} } // end namespace
