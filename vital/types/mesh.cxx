/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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

/**
 * \file
 * \brief Implementation of \link kwiver::vital::mesh mesh \endlink class
 */


#include <vital/types/mesh.h>
#include <vital/logger/logger.h>
#include <Eigen/Geometry>

#include <map>

namespace kwiver {
namespace vital {


/// compute the vector normal to the plane defined by 3 vertices
vector_3d
mesh_tri_normal(const vector_3d& a,
                const vector_3d& b,
                const vector_3d& c)
{
  vector_3d ac(c-a);
  vector_3d ab(b-a);
  return ab.cross(ac);
}



//-----------------------------------------------------------------------------
// Mesh faces

/// Return the group name for a given face index
std::string
mesh_face_array_base
::group_name(unsigned int f) const
{
  if (groups_.empty())
  {
    return "";
  }

  unsigned int i=0;
  for (; i<groups_.size() && groups_[i].second<f; ++i)
  {
  }

  if (i>=groups_.size())
  {
    return "";
  }

  return groups_[i].first;
}


/// Return a set of all faces in a group
std::set<unsigned int>
mesh_face_array_base
::group_face_set(const std::string& name) const
{
  std::set<unsigned int> face_set;
  unsigned int start = 0, end;
  for (unsigned int g=0; g<groups_.size(); ++g)
  {
    end = groups_[g].second;
    if (groups_[g].first == name)
    {
      for (unsigned int i=start; i<end; ++i)
      {
        face_set.insert(i);
      }
    }
    start = end;
  }
  return face_set;
}


/// Assign a group name to all faces currently unnamed
unsigned int
mesh_face_array_base
::make_group(const std::string& name)
{
  unsigned int start_idx = 0;
  if (!groups_.empty())
  {
    start_idx = groups_.back().second;
  }

  if (start_idx < this->size())
  {
    groups_.push_back(std::pair<std::string,unsigned int>(name,this->size()));
  }

  return this->size() - start_idx;
}


/// Append this array of faces (must be the same type)
void
mesh_face_array_base
::append(const mesh_face_array_base& other, unsigned int )
{
  if (this->has_normals() && other.has_normals())
  {
    normals_.insert(normals_.end(), other.normals_.begin(), other.normals_.end());
  }
  else
  {
    normals_.clear();
  }

  if (other.has_groups())
  {
    // group any ungrouped faces in this array
    this->make_group("ungrouped");
    const unsigned int offset = this->size();
    for (unsigned int g=0; g<other.groups_.size(); ++g)
    {
      groups_.push_back(other.groups_[g]);
      groups_.back().second += offset;
    }
  }
}


/// Append this array of faces
void
mesh_face_array
::append(const mesh_face_array_base& other, unsigned int ind_shift)
{
  mesh_face_array_base::append(other,ind_shift);

  const unsigned int new_begin = static_cast<unsigned int>(faces_.size());

  if (other.regularity() == 0)
  {
    const mesh_face_array& fs = static_cast<const mesh_face_array&>(other);
    faces_.insert(faces_.end(), fs.faces_.begin(), fs.faces_.end());

    if (ind_shift > 0)
    {
      for (unsigned int i=new_begin; i<faces_.size(); ++i)
      {
        std::vector<unsigned int>& f = faces_[i];
        for (unsigned int j=0; j<f.size(); ++j)
        {
          f[j] += ind_shift;
        }
      }
    }
  }
  else
  {
    for (unsigned int i=0; i<other.size(); ++i)
    {
      std::vector<unsigned int> f(other.num_verts(i));
      for (unsigned int j=0; j<other.num_verts(i); ++j)
      {
        f[j] = other(i,j) + ind_shift;
      }
      faces_.push_back(f);
    }
  }
}


/// Merge the two face arrays
std::unique_ptr<mesh_face_array_base>
merge_face_arrays(const mesh_face_array_base& f1,
                  const mesh_face_array_base& f2,
                  unsigned int ind_shift)
{
  std::unique_ptr<mesh_face_array_base> f;
  // if both face sets are regular with the same number of vertices per face
  if (f1.regularity() == f2.regularity() || f1.regularity() == 0)
  {
    f.reset(f1.clone());
  }
  else
  {
    f.reset(new mesh_face_array(f1));
  }
  f->append(f2,ind_shift);
  return f;
}


//-----------------------------------------------------------------------------
// Mesh edges


/// Construct from a face index list
mesh_half_edge_set
::mesh_half_edge_set(const std::vector<std::vector<unsigned int> >& face_list)
{
  build_from_ifs(face_list);
}


/// Build the half edges from an indexed face set
void
mesh_half_edge_set
::build_from_ifs(const std::vector<std::vector<unsigned int> >& face_list)
{
  half_edges_.clear();
  typedef std::pair<unsigned int, unsigned int> vert_pair;
  std::map<vert_pair, unsigned int> edge_map;

  face_to_he_.resize(face_list.size(), mesh_invalid_idx);

  unsigned int max_v = 0;

  const unsigned int num_faces = static_cast<unsigned int>(face_list.size());
  for (unsigned int f=0; f<num_faces; ++f)
  {
    const std::vector<unsigned int>& verts = face_list[f];
    const unsigned int num_verts = static_cast<unsigned int>(verts.size());
    unsigned int first_e = mesh_invalid_idx; // first edge index
    unsigned int prev_e = mesh_invalid_idx; // previous edge index
    for (unsigned int i=0; i<num_verts; ++i)
    {
      const unsigned int& v = verts[i];
      if (v > max_v)
      {
        max_v = v;
      }
      unsigned int ni = (i+1) % num_verts;
      const unsigned int& nv = verts[ni];

      vert_pair vp(v,nv);
      if (v > nv)
      {
        vp = vert_pair(nv,v);
      }
      std::map<vert_pair, unsigned int>::iterator m = edge_map.find(vp);
      unsigned int curr_e;
      if (m == edge_map.end())
      {
        curr_e = static_cast<unsigned int>(half_edges_.size());
        edge_map.insert(std::pair<vert_pair, unsigned int>(vp, curr_e));
        half_edges_.push_back(mesh_half_edge(curr_e, mesh_invalid_idx, v, f));
        half_edges_.push_back(mesh_half_edge(curr_e+1, mesh_invalid_idx, nv, mesh_invalid_idx));
      }
      else
      {
        curr_e = m->second + 1;
        assert(half_edges_[curr_e].next_index() == mesh_invalid_idx);
        assert(half_edges_[curr_e].vert_index() == v);
        half_edges_[curr_e].face_ = f;
      }
      if (first_e == mesh_invalid_idx)
      {
        first_e = curr_e;
      }
      if (prev_e != mesh_invalid_idx)
      {
        half_edges_[prev_e].next_ = curr_e;
      }
      prev_e = curr_e;
    }
    if (prev_e != mesh_invalid_idx)
    {
      half_edges_[prev_e].next_ = first_e;
    }
    face_to_he_[f] = first_e;
  }

  vert_to_he_.resize(max_v+1, mesh_invalid_idx);

  // create half edges for boundaries
  for (unsigned int i=0; i<half_edges_.size(); ++i)
  {
    mesh_half_edge& he = half_edges_[i];
    if (i < vert_to_he_[he.vert_index()])
    {
      vert_to_he_[he.vert_index()] = i;
    }
    if (he.next_index() != mesh_invalid_idx)
    {
      continue;
    }
    unsigned int next_b = half_edges_[i].pair_index();
    while(half_edges_[next_b].face_index() != mesh_invalid_idx)
    {
      f_iterator fi(next_b,*this);
      while (fi->next_index() != next_b)
      {
        ++fi;
      }
      next_b = fi->pair_index();
    }
    he.next_ = next_b;
  }
}


/// Count the number of vertices pointed to by these edges
unsigned int
mesh_half_edge_set
::num_verts() const
{
  unsigned int count = 0;
  for (unsigned int i=0; i<vert_to_he_.size(); ++i)
  {
    if (vert_to_he_[i] != mesh_invalid_idx)
    {
      ++count;
    }
  }
  return count;
}


/// Count the number of faces pointed to by these edges
unsigned int
mesh_half_edge_set
::num_faces() const
{
  unsigned int count = 0;
  for (unsigned int i=0; i<face_to_he_.size(); ++i)
  {
    if (face_to_he_[i] != mesh_invalid_idx)
    {
      ++count;
    }
  }
  return count;
}


//-----------------------------------------------------------------------------
// Mesh


/// Copy Constructor
mesh
::mesh(const mesh& other)
  : verts_((other.verts_.get()) ? other.verts_->clone() : 0),
    faces_((other.faces_.get()) ? other.faces_->clone() : 0),
    half_edges_(other.half_edges_),
    tex_coords_(other.tex_coords_),
    tex_source_(other.tex_source_),
    valid_tex_faces_(other.valid_tex_faces_),
    tex_coord_status_(other.tex_coord_status_)
{
}


/// Assignment operator
mesh&
mesh
::operator=(mesh const& other)
{
  if (this != &other)
  {
    verts_ = std::unique_ptr<mesh_vertex_array_base>((other.verts_.get()) ?
                                                   other.verts_->clone() : 0);
    faces_ = std::unique_ptr<mesh_face_array_base>((other.faces_.get()) ?
                                                 other.faces_->clone() : 0);
    half_edges_ = other.half_edges_;
    tex_coords_ = other.tex_coords_;
    valid_tex_faces_ = other.valid_tex_faces_;
    tex_coord_status_ = other.tex_coord_status_;
  }
  return *this;
}


/// Merge the data from another mesh into this one
void
mesh
::merge(const mesh& other)
{
  const unsigned num_v = this->num_verts();
  const unsigned num_e = this->num_edges();
  faces_ = merge_face_arrays(*this->faces_,*other.faces_,verts_->size());
  verts_->append(*other.verts_);

  if (this->has_tex_coords() == TEX_COORD_NONE)
  {
    std::vector<vector_2d> tex;
    if (other.has_tex_coords() == TEX_COORD_ON_VERT)
    {
      tex = std::vector<vector_2d>(num_v, vector_2d(0,0));
    }
    else if (other.has_tex_coords() == TEX_COORD_ON_CORNER)
    {
      unsigned int nb_corners = faces_->regularity() * num_faces();
      if (nb_corners == 0)
      {
        for (unsigned int f = 0; f < this->num_faces(); ++f)
        {
            nb_corners += this->faces().num_verts(f);
        }
      }
      tex = std::vector<vector_2d>(nb_corners, vector_2d(0, 0));
    }

    tex.insert(tex.end(), other.tex_coords().begin(), other.tex_coords().end());
    this->set_tex_coords(tex);
  }
  else if (this->has_tex_coords() == other.has_tex_coords())
  {
    this->tex_coords_.insert(this->tex_coords_.end(),
                             other.tex_coords().begin(),
                             other.tex_coords().end());
  }

  if (this->has_half_edges() && other.has_half_edges())
  {
    this->build_edge_graph();
  }
  else
  {
    this->half_edges_.clear();
  }
}


/// Set the texture coordinates
void
mesh
::set_tex_coords(const std::vector<vector_2d>& tc)
{
  unsigned int nb_corners = faces_->regularity() * num_faces();
  if (nb_corners == 0)
  {
    for (unsigned int f = 0; f < this->num_faces(); ++f)
    {
        nb_corners += this->faces().num_verts(f);
    }
  }

  if (tc.size() == this->num_verts())
  {
    tex_coord_status_ = TEX_COORD_ON_VERT;
  }
  else if (tc.size() == nb_corners)
  {
    tex_coord_status_ = TEX_COORD_ON_CORNER;
  }
  else
  {
    tex_coord_status_ = TEX_COORD_NONE;
  }

  tex_coords_ = tc;
}


/// Construct the half edges graph structure
void
mesh
::build_edge_graph()
{
  const mesh_face_array_base& faces = this->faces();
  std::vector<std::vector<unsigned int> > face_list(faces.size());
  for (unsigned int f=0; f<faces.size(); ++f)
  {
    face_list[f].resize(faces.num_verts(f));
    for (unsigned int v=0; v<faces.num_verts(f); ++v)
    {
      face_list[f][v] =  faces(f,v);
    }
  }

  half_edges_.build_from_ifs(face_list);
}


/// Compute vertex normals
void
mesh
::compute_vertex_normals()
{
  if (!this->has_half_edges())
  {
    this->build_edge_graph();
  }

  const mesh_half_edge_set& half_edges = this->half_edges();
  mesh_vertex_array<3>& verts = this->vertices<3>();

  std::vector<vector_3d> normals(this->num_verts(), vector_3d(0,0,0));

  for (unsigned int he=0; he < half_edges.size(); ++he)
  {
    mesh_half_edge_set::f_const_iterator fi(he,half_edges);
    if (fi->is_boundary())
    {
      continue;
    }
    const unsigned int vp = fi->vert_index();
    const unsigned int v = (++fi)->vert_index();
    const unsigned int vn = (++fi)->vert_index();
    normals[v] += mesh_tri_normal(verts[v],verts[vn],verts[vp]).normalized();
  }

  for (unsigned v=0; v<verts.size(); ++v)
  {
    normals[v].normalize();
  }

  verts.set_normals(normals);
}


/// Compute vertex normals using face normals
void
mesh
::compute_vertex_normals_from_faces()
{
  if (!this->has_half_edges())
  {
    this->build_edge_graph();
  }

  if (!this->faces_->has_normals())
  {
    this->compute_face_normals();
  }

  const std::vector<vector_3d>& fnormals = faces_->normals();

  const mesh_half_edge_set& half_edges = this->half_edges();
  mesh_vertex_array<3>& verts = this->vertices<3>();

  std::vector<vector_3d> normals(this->num_verts(), vector_3d(0,0,0));

  for (unsigned int he=0; he < half_edges.size(); ++he)
  {
    const mesh_half_edge& half_edge = half_edges[he];
    if (half_edge.is_boundary())
    {
      continue;
    }
    const unsigned int v = half_edge.vert_index();
    normals[v] += fnormals[half_edge.face_index()].normalized();
  }

  for (unsigned v=0; v<verts.size(); ++v)
  {
    normals[v].normalize();
  }

  verts.set_normals(normals);
}


/// Compute face normals
void
mesh
::compute_face_normals(bool norm)
{
  mesh_face_array_base& faces = this->faces();
  const mesh_vertex_array<3>& verts = this->vertices<3>();

  std::vector<vector_3d> normals(this->num_faces(), vector_3d(0,0,0));

  for (unsigned int i=0; i<faces.size(); ++i)
  {
    const unsigned int num_v = faces.num_verts(i);
    vector_3d& n = normals[i];
    for (unsigned int j=2; j<num_v; ++j)
    {
      n += mesh_tri_normal(verts[faces(i,0)],
                            verts[faces(i,j-1)],
                            verts[faces(i,j)]);
    }
    if (norm)
    {
      n.normalize();
    }
  }

  faces.set_normals(normals);
}


/// Map a barycentric coordinate (u,v) on triangle \param tri into texture space
vector_2d
mesh
::texture_map(unsigned int tri, double u, double v) const
{
  vector_2d tex(0,0);
  if (this->tex_coord_status_ == TEX_COORD_ON_VERT)
  {
    const unsigned int v1 = (*faces_)(tri,0);
    const unsigned int v2 = (*faces_)(tri,1);
    const unsigned int v3 = (*faces_)(tri,2);
    tex += (1-u-v) * tex_coords_[v1];
    tex += u * tex_coords_[v2];
    tex += v * tex_coords_[v3];
  }
  else if (this->tex_coord_status_ == TEX_COORD_ON_CORNER)
  {
    const unsigned int i1 = 3 * tri + 0;
    const unsigned int i2 = 3 * tri + 1;
    const unsigned int i3 = 3 * tri + 2;
    tex += (1 - u - v) * tex_coords_[i1];
    tex += u * tex_coords_[i2];
    tex += v * tex_coords_[i3];
  }
  return tex;
}


/// Set the vector indicating which faces have texture
void
mesh
::set_valid_tex_faces(const std::vector<bool>& valid)
{
  if (valid.size() == this->num_faces() && has_tex_coords())
  {
    valid_tex_faces_ = valid;
  }
}


/// Label all faces with positive (counter clockwise orientation) area as valid
void
mesh
::label_ccw_tex_faces_valid()
{
  switch (tex_coord_status_)
  {
    case TEX_COORD_ON_VERT:
    {
      valid_tex_faces_.resize(this->num_faces());
      mesh_face_array_base& faces = this->faces();
      for (unsigned int f=0; f<num_faces(); ++f)
      {
        const unsigned int num_v = faces.num_verts(f);
        // compute (2x) area with Green's theorem
        double area = 0.0;
        for (unsigned int i=0, j=num_v-1; i<num_v; j=i++)
        {
          const vector_2d& tex_i = tex_coords_[faces(f,i)];
          const vector_2d& tex_j = tex_coords_[faces(f,j)];
          area += tex_j.x() * tex_i.y() - tex_i.x() * tex_j.y();
        }
        valid_tex_faces_[f] = area > 0;
      }
      break;
    }
    case TEX_COORD_ON_CORNER:
    {
      logger_handle_t logger(get_logger( "vital.mesh" ));
      LOG_ERROR(logger, "mesh::label_ccw_tex_faces_valid()"
                        " not implemented for TEX_COORD_ON_CORNER");
      break;
    }
    default:
      break;
  }
}

} // end namespace vital
} // end namespace kwiver
