/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "category_hierarchy.h"

#include <stdexcept>
#include <algorithm>

namespace kwiver {
namespace vital {


// -----------------------------------------------------------------------------
category_hierarchy
::category_hierarchy()
{
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
  for( std::map< label_t, category* >::const_iterator p = m_hierarchy.begin();
       p != m_hierarchy.end(); p++ )
  {
    delete p->second;
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

  category* new_entry = new category();
  m_hierarchy[ class_name ] = new_entry;

  new_entry->category_name = class_name;
  new_entry->category_id = id;

  if( !parent_name.empty() )
  {
    std::map< label_t, category* >::const_iterator itr = find( parent_name );
    new_entry->parents.push_back( itr->second );
  }
}


// -----------------------------------------------------------------------------
category_hierarchy::label_id_t
category_hierarchy
::get_class_id( const label_t& class_name ) const
{
  std::map< label_t, category* >::const_iterator itr = this->find( class_name );

  return itr->second->category_id;
}


// -----------------------------------------------------------------------------
category_hierarchy::label_vec_t
category_hierarchy
::get_class_parents( const label_t& class_name ) const
{
  label_vec_t output;

  std::map< label_t, category* >::const_iterator itr = this->find( class_name );

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
  std::map< label_t, category* >::const_iterator itr1 = this->find( child_name );
  std::map< label_t, category* >::const_iterator itr2 = this->find( parent_name );

  itr1->second->parents.push_back( itr2->second );
  itr2->second->children.push_back( itr1->second );
}


// -----------------------------------------------------------------------------
category_hierarchy::label_vec_t
category_hierarchy
::class_names() const
{
  label_vec_t names;

  for( std::map< label_t, category* >::const_iterator p = m_hierarchy.begin();
       p != m_hierarchy.end(); p++ )
  {
    names.push_back( p->first );
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
std::map< category_hierarchy::label_t, category_hierarchy::category* >::const_iterator
category_hierarchy
::find( const label_t& lbl ) const
{
  std::map< label_t, category* >::const_iterator itr = m_hierarchy.find( lbl );

  if( itr == m_hierarchy.end() )
  {
    throw std::runtime_error( "Class node does not exist." );
  }

  return itr;
}


} } // end namespace
