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
 * \brief Header for \link kwiver::vital::mesh mesh \endlink and
 *        related classes.
 *
 * This indexed mesh implementation is based on imesh from VXL.
 */

#ifndef VITAL_MESH_H_
#define VITAL_MESH_H_


#include <vector>
#include <set>
#include <memory>

#include <vital/vital_export.h>
#include <vital/types/vector.h>


namespace kwiver {
namespace vital {


//-----------------------------------------------------------------------------
// Mesh vertices

/// Abstract base class for a collection of vertices
class mesh_vertex_array_base
{
public:
  /// Destructor
  virtual ~mesh_vertex_array_base() {}

  /// returns the number of vertices
  virtual unsigned int size() const = 0;

  /// returns the dimension of the vertices
  virtual unsigned int dim() const = 0;

  /// Access a vertex coordinate by vertex index and coordinate index
  virtual double operator() (unsigned int v, unsigned int i) const = 0;

  /// Produce a clone of this object (dynamic copy)
  virtual mesh_vertex_array_base* clone() const = 0;

  /// Append these vertices (assuming the same type)
  virtual void append(const mesh_vertex_array_base& verts)
  {
    if (this->has_normals() && verts.has_normals())
    {
      normals_.insert(normals_.end(), verts.normals_.begin(), verts.normals_.end());
    }
    else
    {
      normals_.clear();
    }
  }

  /// Return true if the vertices have normals
  bool has_normals() const { return !normals_.empty(); }

  /// Set the vertex normals
  void set_normals(const std::vector<vector_3d>& n)
  {
    assert(n.size() == this->size());
    normals_ = n;
  }

  /// Access a vertex normal
  vector_3d& normal(unsigned int v) { return normals_[v]; }
  const vector_3d& normal(unsigned int v) const { return normals_[v]; }

  /// Access the normals
  const std::vector<vector_3d>& normals() const { return normals_; }

protected:
  std::vector<vector_3d> normals_;
};


/// An array of vertices of dimension d
template <unsigned int d>
class mesh_vertex_array : public mesh_vertex_array_base
{
  /// typedef for d-dimensional point
  /**
   * \note Eigen::Matrix<double,2,1> == vector_2d
   *   and Eigen::Matrix<double,3,1> == vector_3d
   */
  typedef Eigen::Matrix<double,3,1> vert_t;

  /// vector of d-dimensional points
  std::vector<vert_t> verts_;

public:
  /// Default Constructor
  mesh_vertex_array<d>() {}

  /// Constructor (from size)
  mesh_vertex_array<d>(unsigned int size)
  : verts_(size) {}

  /// Constructor (from vector)
  mesh_vertex_array<d>(const std::vector<vert_t>& verts)
  : verts_(verts) {}

  /// Produce a clone of this object (dynamic copy)
  virtual mesh_vertex_array_base* clone() const
  {
    return new mesh_vertex_array<d>(*this);
  }

  /// returns the number of vertices
  virtual unsigned int size() const { return static_cast<unsigned int>(verts_.size()); }

  /// returns the dimension of the vertices
  virtual unsigned int dim() const { return d; }

  /// Access a vertex coordinate by vertex index and coordinate index
  virtual double operator() (unsigned int v, unsigned int i) const { return verts_[v][i]; }

  /// Append these vertices (assuming the same type)
  virtual void append(const mesh_vertex_array_base& verts)
  {
    assert(verts.dim() == d);
    const mesh_vertex_array<d>& v = static_cast<const mesh_vertex_array<d>&>(verts);
    verts_.insert(verts_.end(), v.verts_.begin(), v.verts_.end());
    mesh_vertex_array_base::append(verts);
  }

  /// Add a vertex to the array
  void push_back(const vert_t& v) { verts_.push_back(v); }

  /// Access a vertex
  vert_t& operator[] (unsigned int v) { return verts_[v]; }
  const vert_t& operator[] (unsigned int v) const { return verts_[v]; }

  //=====================================================
  // Vertex Iterators
  typedef typename std::vector<vert_t>::iterator iterator;
  typedef typename std::vector<vert_t>::const_iterator const_iterator;

  iterator begin() { return verts_.begin(); }
  const_iterator begin() const { return verts_.begin(); }

  iterator end() { return verts_.end(); }
  const_iterator end() const { return verts_.end(); }
};


/// compute the vector normal to the plane defined by 3 vertices
VITAL_EXPORT
vector_3d mesh_tri_normal(const vector_3d& a,
                          const vector_3d& b,
                          const vector_3d& c);



//-----------------------------------------------------------------------------
// Mesh faces

/// The special value used to indicate and invalid index
const unsigned int mesh_invalid_idx = static_cast<unsigned int>(-1);


/// A mesh face with a fixed number of vertices
template <unsigned s>
class mesh_regular_face
{
public:
  /// Default Constructor
  mesh_regular_face()
  {
    std::fill_n(verts_, s, mesh_invalid_idx);
  }

  /// Constructor from a vector
  mesh_regular_face(const std::vector<unsigned int>& verts)
  {
    assert(verts.size()==s);
    std::copy(verts.begin(), verts.end(), verts_);
  }

  /// return the number of vertices
  unsigned int num_verts() const { return s; }

  void flip_orientation()
  {
    std::reverse(verts_,verts_+s);
  }

  /// Accessor
  unsigned int operator[] (unsigned int i) const { return verts_[i]; }
  unsigned int& operator[] (unsigned int i) { return verts_[i]; }
protected:
  unsigned int verts_[s];
};


/// A triangle face
class mesh_tri : public mesh_regular_face<3>
{
public:
  mesh_tri(unsigned int a, unsigned int b, unsigned int c)
  {
    verts_[0] = a;
    verts_[1] = b;
    verts_[2] = c;
  }
};

/// A quadrilateral face
class mesh_quad : public mesh_regular_face<4>
{
public:
  mesh_quad(unsigned int a, unsigned int b,
             unsigned int c, unsigned int d)
  {
    verts_[0] = a;
    verts_[1] = b;
    verts_[2] = c;
    verts_[3] = d;
  }
};


/// Abstract base class for a collection of faces
class VITAL_EXPORT mesh_face_array_base
{
public:
  /// Destructor
  virtual ~mesh_face_array_base() {}

  /// returns the number of vertices per face if the same for all faces, zero otherwise
  virtual unsigned int regularity() const = 0;

  /// returns the number of faces
  virtual unsigned int size() const = 0;

  /// returns the number of vertices in face \param f
  virtual unsigned int num_verts(unsigned int f) const = 0;

  /// Access a vertex index by face index and within-face index
  virtual unsigned int operator() (unsigned int f, unsigned int i) const = 0;

  /// Flip a face over, inverting its orientation
  virtual void flip_orientation (unsigned int f)
  {
    if (has_normals())
    {
      normals_[f] *= -1;
    }
  }

  /// Produce a clone of this object (dynamic copy)
  virtual mesh_face_array_base* clone() const = 0;

  /// Append this array of faces (must be the same type)
  /**
   * Optionally shift the indices in \param other by \param ind_shift
   */
  virtual void append(const mesh_face_array_base& other,
                      unsigned int ind_shift=0);

  /// Return true if the faces have normals
  bool has_normals() const { return !normals_.empty(); }

  /// Set the face normals
  void set_normals(const std::vector<vector_3d >& n)
  {
    assert(n.size() == this->size());
    normals_ = n;
  }

  /// Access a face normal
  vector_3d& normal(unsigned int f) { return normals_[f]; }
  const vector_3d& normal(unsigned int f) const { return normals_[f]; }

  /// Access the entire vector of normals
  const std::vector<vector_3d >& normals() const { return normals_; }

  /// Returns true if the faces have named groups
  bool has_groups() const { return !groups_.empty(); }

  /// Return the group name for a given face index
  std::string group_name(unsigned int f) const;

  /// Assign a group name to all faces currently unnamed
  /**
   * Return the number of faces in the new group
   */
  unsigned int make_group(const std::string& name);

  /// Return a set of all faces in a group
  std::set<unsigned int> group_face_set(const std::string& name) const;

  /// Access the groups
  const std::vector<std::pair<std::string,unsigned int> >& groups() const { return groups_; }

protected:
  /// named groups of adjacent faces (a partition of the face array)
  /**
   * Integers mark the group's ending vertex + 1
   */
  std::vector<std::pair<std::string,unsigned int> > groups_;

  /// vectors that are normal to each face
  std::vector<vector_3d > normals_;
};


/// An array of mesh faces of arbitrary size
class VITAL_EXPORT mesh_face_array : public mesh_face_array_base
{
  std::vector<std::vector<unsigned int> > faces_;

public:
  /// Default Constructor
  mesh_face_array() {}

  /// Constructor
  mesh_face_array(unsigned int size) : faces_(size) {}

  /// Constructor (from a vector)
  mesh_face_array(const std::vector<std::vector<unsigned int> >& faces)
  : faces_(faces) {}

  /// Copy Constructor
  mesh_face_array(const mesh_face_array& other)
  : mesh_face_array_base(other), faces_(other.faces_) {}

  /// Construct from base class
  explicit mesh_face_array(const mesh_face_array_base& fb)
  : mesh_face_array_base(fb), faces_(fb.size())
  {
    for (unsigned int i=0; i<fb.size(); ++i)
    {
      for (unsigned int j=0; j<fb.num_verts(i); ++j)
      {
        faces_[i].push_back(fb(i,j));
      }
    }
  }

  /// returns the number of vertices per face if the same for all faces, zero otherwise
  virtual unsigned int regularity() const { return 0; }

  /// returns the number of faces
  virtual unsigned int size() const { return static_cast<unsigned int>(faces_.size()); }

  /// returns the number of vertices in face \param f
  virtual unsigned int num_verts(unsigned int f) const { return static_cast<unsigned int>(faces_[f].size()); }

  /// Access a vertex index by face index and within-face index
  virtual unsigned int operator() (unsigned int f, unsigned int i) const { return faces_[f][i]; }

  /// Flip a face over, inverting its orientation
  virtual void flip_orientation (unsigned int f)
  {
    std::reverse(faces_[f].begin(),faces_[f].end());
    mesh_face_array_base::flip_orientation(f);
  }

  /// Produce a clone of this object (dynamic copy)
  virtual mesh_face_array_base* clone() const
  {
    return new mesh_face_array(*this);
  }

  /// Append this array of faces
  /**
   * Optionally shift the indices in \param other by \param ind_shift
   */
  virtual void append(const mesh_face_array_base& other,
                      unsigned int ind_shift=0);

  /// Add a face to the array
  void push_back(const std::vector<unsigned int>& f) { faces_.push_back(f); }

  /// Add a face to the array
  template <unsigned int s>
  void push_back(const mesh_regular_face<s>& f)
  {
    std::vector<unsigned int> f2(s);
    for (unsigned int i=0; i<s; ++i)
    {
      f2[i] = f[i];
    }
    this->push_back(f2);
  }

  /// Access face \param f
  std::vector<unsigned int>& operator[] (unsigned int f) { return faces_[f]; }
  const std::vector<unsigned int>& operator[] (unsigned int f) const { return faces_[f]; }
};


/// An array of mesh faces of arbitrary size
template <unsigned int s>
class mesh_regular_face_array : public mesh_face_array_base
{
  std::vector<mesh_regular_face<s> > faces_;

public:
  /// Default Constructor
  mesh_regular_face_array<s>() {}

  /// Constructor
  mesh_regular_face_array<s>(unsigned int size) : faces_(size) {}

  /// Constructor (from a vector)
  mesh_regular_face_array<s>(const std::vector<mesh_regular_face<s> >& faces) : faces_(faces) {}

  /// returns the number of vertices per face if the same for all faces
  /**
   * Returns zero if the number of vertices may vary from face to face.
   */
  virtual unsigned int regularity() const { return s; }

  /// returns the number of faces
  virtual unsigned int size() const { return static_cast<unsigned int>(faces_.size()); }

  /// returns the number of vertices in face \param f
  virtual unsigned int num_verts(unsigned int /*f*/) const { return s; }

  /// Access a vertex index by face index and within-face index
  virtual unsigned int operator() (unsigned int f, unsigned int i) const { return faces_[f][i]; }

  /// Flip a face over, inverting its orientation
  virtual void flip_orientation (unsigned int f)
  {
    faces_[f].flip_orientation();
    mesh_face_array_base::flip_orientation(f);
  }

  /// Produce a clone of this object (dynamic copy)
  virtual mesh_face_array_base* clone() const
  {
    return new mesh_regular_face_array<s>(*this);
  }

  /// Append this array of faces (must be the same type)
  /**
   * Optionally shift the indices in \param other by \param ind_shift
   */
  virtual void append(const mesh_face_array_base& other,
                      unsigned int ind_shift=0)
  {
    mesh_face_array_base::append(other,ind_shift);
    assert(other.regularity() == s);
    const mesh_regular_face_array<s>& fs =
        static_cast<const mesh_regular_face_array<s>&>(other);
    const unsigned int new_begin = static_cast<unsigned int>(faces_.size());
    faces_.insert(faces_.end(), fs.faces_.begin(), fs.faces_.end());
    if (ind_shift > 0)
    {
      for (unsigned int i=new_begin; i<faces_.size(); ++i)
      {
        mesh_regular_face<s>& f = faces_[i];
        for (unsigned int j=0; j<s; ++j)
        {
          f[j] += ind_shift;
        }
      }
    }
  }

  /// Add a face to the array
  void push_back(const mesh_regular_face<s>& f) { faces_.push_back(f); }

  /// Access face \param f
  mesh_regular_face<s>& operator[] (unsigned int f) { return faces_[f]; }
  const mesh_regular_face<s>& operator[] (unsigned int f) const { return faces_[f]; }

  //=====================================================
  // Face Iterators
  typedef typename std::vector<mesh_regular_face<s> >::iterator iterator;
  typedef typename std::vector<mesh_regular_face<s> >::const_iterator const_iterator;

  iterator begin() { return faces_.begin(); }
  const_iterator begin() const { return faces_.begin(); }

  iterator end() { return faces_.end(); }
  const_iterator end() const { return faces_.end(); }
};


/// Merge the two face arrays
/**
 * Shift the mesh indices in \param f2 by \param ind_shift
 */
VITAL_EXPORT
std::unique_ptr<mesh_face_array_base>
merge_face_arrays(const mesh_face_array_base& f1,
                  const mesh_face_array_base& f2,
                  unsigned int ind_shift=0);



//-----------------------------------------------------------------------------
// Mesh edges

class VITAL_EXPORT mesh_half_edge
{
  friend class mesh_half_edge_set;
public:
  mesh_half_edge(unsigned int e, unsigned int n, unsigned int v, unsigned int f)
  : next_(n), edge_(e), vert_(v), face_(f) {}

  /// return the next half-edge index
  unsigned int next_index() const { return next_; }
  /// return the pair half-edge index
  unsigned int pair_index() const { return edge_^1; }

  /// return the index of the full edge
  unsigned int edge_index() const { return edge_>>1; }
  /// return the index of the half-edge
  unsigned int half_edge_index() const { return edge_; }

  /// return the vertex index
  unsigned int vert_index() const { return vert_; }
  /// return the face index
  unsigned int face_index() const { return face_; }

  bool is_boundary() const { return face_ == mesh_invalid_idx; }

private:
  unsigned int next_;
  unsigned int edge_;

  unsigned int vert_;
  unsigned int face_;
};


/// A collection of indexed half edges
class VITAL_EXPORT mesh_half_edge_set
{
public:
  /// Default Constructor
  mesh_half_edge_set() {}
  /// Construct from a face index list
  mesh_half_edge_set(const std::vector<std::vector<unsigned int> >& face_list);

  /// Build the half edges from an indexed face set
  void build_from_ifs(const std::vector<std::vector<unsigned int> >& face_list);

  /// Access by index
  const mesh_half_edge& operator [] (unsigned int i) const { return half_edges_[i]; }
  /// Access by index
  mesh_half_edge& operator [] (unsigned int i)  { return half_edges_[i]; }

  /// number of half edges
  unsigned int size() const { return static_cast<unsigned int>(half_edges_.size()); }

  /// clear the edges
  void clear()
  {
    half_edges_.clear();
    vert_to_he_.clear();
    face_to_he_.clear();
  }

  // forward declare iterators
  class f_iterator;
  class f_const_iterator;
  class v_iterator;
  class v_const_iterator;

  //=====================================================
  // Mesh Face Iterators - each half edge touches the same face

  /// An iterator of half edges adjacent to a face
  class f_iterator : public std::iterator<std::forward_iterator_tag,mesh_half_edge>
  {
    friend class f_const_iterator;
    friend class v_iterator;
  public:
    /// Constructor
    f_iterator(unsigned int hei, mesh_half_edge_set& edge_set)
      :half_edge_index_(hei), edge_set_(edge_set) {}

    /// Constructor from vertex iterator
    explicit f_iterator(const v_iterator& other)
      :half_edge_index_(other.half_edge_index_), edge_set_(other.edge_set_) {}

    /// Assignment
    f_iterator& operator = (const f_iterator& other)
    {
      if (this != &other)
      {
        assert(&edge_set_ == &other.edge_set_);
        half_edge_index_ = other.half_edge_index_;
      }
      return *this;
    }

    mesh_half_edge & operator*() const { return edge_set_[half_edge_index_]; }
    mesh_half_edge * operator->() const { return &**this; }
    mesh_half_edge & pair() const { return edge_set_[half_edge_index_^1]; }
    f_iterator & operator++ () // pre-inc
    {
      half_edge_index_ = edge_set_[half_edge_index_].next_index();
      return *this;
    }
    f_iterator operator++(int) // post-inc
    {
      f_iterator old = *this;
      ++*this;
      return old;
    }

    bool operator == (const f_iterator& other) const
    {
      return this->half_edge_index_ == other.half_edge_index_ &&
            &(this->edge_set_) == &(other.edge_set_);
    }

    bool operator != (const f_iterator& other) const
    {
      return !(*this == other);
    }

    bool operator == (const f_const_iterator& other) const
    {
      return this->half_edge_index_ == other.half_edge_index_ &&
            &(this->edge_set_) == &(other.edge_set_);
    }

    bool operator != (const f_const_iterator& other) const
    {
      return !(*this == other);
    }

  private:
    unsigned int half_edge_index_;
    mesh_half_edge_set& edge_set_;
  };

  /// A const iterator of half edges adjacent to a face
  class f_const_iterator : public std::iterator<std::forward_iterator_tag,mesh_half_edge>
  {
    friend class f_iterator;
    friend class v_const_iterator;
  public:
    /// Constructor
    f_const_iterator(unsigned int hei, const mesh_half_edge_set& edge_set)
      :half_edge_index_(hei), edge_set_(edge_set) {}

    /// Constructor from non-const iterator
    f_const_iterator(const f_iterator& other)
      :half_edge_index_(other.half_edge_index_), edge_set_(other.edge_set_) {}

    /// Constructor from vertex iterator
    explicit f_const_iterator(const v_const_iterator& other)
      :half_edge_index_(other.half_edge_index_), edge_set_(other.edge_set_) {}

    /// Assignment
    f_const_iterator& operator = (const f_const_iterator& other)
    {
      if (this != &other)
      {
        assert(&edge_set_ == &other.edge_set_);
        half_edge_index_ = other.half_edge_index_;
      }
      return *this;
    }

    const mesh_half_edge & operator*() const { return edge_set_[half_edge_index_]; }
    const mesh_half_edge * operator->() const { return &**this; }
    const mesh_half_edge & pair() const { return edge_set_[half_edge_index_^1]; }
    f_const_iterator & operator++ () // pre-inc
    {
      half_edge_index_ = edge_set_[half_edge_index_].next_index();
      return *this;
    }
    f_const_iterator operator++(int) // post-inc
    {
      f_const_iterator old = *this;
      ++*this;
      return old;
    }

    bool operator == (const f_const_iterator& other) const
    {
      return this->half_edge_index_ == other.half_edge_index_ &&
            &(this->edge_set_) == &(other.edge_set_);
    }

    bool operator != (const f_const_iterator& other) const
    {
      return !(*this == other);
    }

  private:
    unsigned int half_edge_index_;
    const mesh_half_edge_set& edge_set_;
  };


  //=====================================================
  // Mesh Vertex Iterators - each half edge touches the same vertex

  /// An iterator of half edges adjacent to a vertex
  class v_iterator : public std::iterator<std::forward_iterator_tag,mesh_half_edge>
  {
    friend class v_const_iterator;
    friend class f_iterator;
  public:
    /// Constructor
    v_iterator(unsigned int hei, mesh_half_edge_set& edge_set)
      :half_edge_index_(hei), edge_set_(edge_set) {}

    /// Constructor from face iterator
    explicit v_iterator(const f_iterator& other)
      :half_edge_index_(other.half_edge_index_), edge_set_(other.edge_set_) {}

    /// Assignment
    v_iterator& operator = (const v_iterator& other)
    {
      if (this != &other)
      {
        assert(&edge_set_ == &other.edge_set_);
        half_edge_index_ = other.half_edge_index_;
      }
      return *this;
    }

    mesh_half_edge & operator*() const { return edge_set_[half_edge_index_]; }
    mesh_half_edge * operator->() const { return &**this; }
    mesh_half_edge & pair() const { return edge_set_[half_edge_index_^1]; }
    v_iterator & operator++ () // pre-inc
    {
      half_edge_index_ = half_edge_index_ ^ 1; // pair index
      half_edge_index_ = edge_set_[half_edge_index_].next_index();
      return *this;
    }
    v_iterator operator++(int) // post-inc
    {
      v_iterator old = *this;
      ++*this;
      return old;
    }

    bool operator == (const v_iterator& other) const
    {
      return this->half_edge_index_ == other.half_edge_index_ &&
            &(this->edge_set_) == &(other.edge_set_);
    }

    bool operator != (const v_iterator& other) const
    {
      return !(*this == other);
    }

    bool operator == (const v_const_iterator& other) const
    {
      return this->half_edge_index_ == other.half_edge_index_ &&
            &(this->edge_set_) == &(other.edge_set_);
    }

    bool operator != (const v_const_iterator& other) const
    {
      return !(*this == other);
    }

  private:
    unsigned int half_edge_index_;
    mesh_half_edge_set& edge_set_;
  };

  /// A const iterator of half edges adjacent to a vertex
  class v_const_iterator : public std::iterator<std::forward_iterator_tag,mesh_half_edge>
  {
    friend class v_iterator;
    friend class f_const_iterator;
  public:
    /// Constructor
    v_const_iterator(unsigned int hei, const mesh_half_edge_set& edge_set)
      :half_edge_index_(hei), edge_set_(edge_set) {}

    /// Constructor from non-const iterator
    v_const_iterator(const v_iterator& other)
      :half_edge_index_(other.half_edge_index_), edge_set_(other.edge_set_) {}

    /// Constructor from face iterator
    explicit v_const_iterator(const f_const_iterator& other)
      :half_edge_index_(other.half_edge_index_), edge_set_(other.edge_set_) {}

    /// Assignment
    v_const_iterator& operator = (const v_const_iterator& other)
    {
      if (this != &other)
      {
        assert(&edge_set_ == &other.edge_set_);
        half_edge_index_ = other.half_edge_index_;
      }
      return *this;
    }

    const mesh_half_edge & operator*() const { return edge_set_[half_edge_index_]; }
    const mesh_half_edge * operator->() const { return &**this; }
    const mesh_half_edge & pair() const { return edge_set_[half_edge_index_^1]; }
    v_const_iterator & operator++ () // pre-inc
    {
      half_edge_index_ = half_edge_index_ ^ 1; // pair index
      half_edge_index_ = edge_set_[half_edge_index_].next_index();
      return *this;
    }
    v_const_iterator operator++(int) // post-inc
    {
      v_const_iterator old = *this;
      ++*this;
      return old;
    }

    bool operator == (const v_const_iterator& other) const
    {
      return this->half_edge_index_ == other.half_edge_index_ &&
            &(this->edge_set_) == &(other.edge_set_);
    }

    bool operator != (const v_const_iterator& other) const
    {
      return !(*this == other);
    }

  private:
    unsigned int half_edge_index_;
    const mesh_half_edge_set& edge_set_;
  };

  /// Access a face iterator for face \param f
  f_const_iterator face_begin(unsigned int f) const { return f_const_iterator(face_to_he_[f],*this); }
  f_iterator face_begin(unsigned int f) { return f_iterator(face_to_he_[f],*this); }

  /// Access a vertex iterator for vertex \param v
  v_const_iterator vertex_begin(unsigned int v) const { return v_const_iterator(vert_to_he_[v],*this); }
  v_iterator vertex_begin(unsigned int v) { return v_iterator(vert_to_he_[v],*this); }

  /// Count the number of vertices pointed to by these edges
  unsigned int num_verts() const;

  /// Count the number of faces pointed to by these edges
  unsigned int num_faces() const;

private:
  std::vector<mesh_half_edge> half_edges_;
  std::vector<unsigned int> vert_to_he_;
  std::vector<unsigned int> face_to_he_;
};



//-----------------------------------------------------------------------------
// Mesh


/// A simple indexed mesh
class VITAL_EXPORT mesh
{
public:
  /// Default Constructor
  mesh() : tex_coord_status_(TEX_COORD_NONE) {}

  /// Constructor from vertex and face arrays
  /**
   * Takes ownership of these arrays
   */
  mesh(std::unique_ptr<mesh_vertex_array_base> verts,
             std::unique_ptr<mesh_face_array_base> faces)
  : verts_(std::move(verts)),
    faces_(std::move(faces)),
    tex_coord_status_(TEX_COORD_NONE) {}

  /// Copy Constructor
  mesh(const mesh& other);

  /// Assignment operator
  mesh& operator=(mesh const& other);

  /// Return the number of vertices
  unsigned int num_verts() const {return verts_->size();}

  /// Return the number of faces
  unsigned int num_faces() const {return faces_->size();}

  /// Return the number of edges
  unsigned int num_edges() const {return half_edges_.size()/2;}

  /// Merge the data from another mesh into this one
  /**
   * Duplicates are not removed
   */
  void merge(const mesh& other);

  /// Return true if the mesh has been initialized
  bool is_init() const { return verts_.get() && faces_.get(); }

  /// Access the vector of vertices
  const mesh_vertex_array_base& vertices() const { return *verts_; }
  mesh_vertex_array_base& vertices() { return *verts_; }

  /// Access the vector of vertices cast to a dimension
  template <unsigned int d>
  const mesh_vertex_array<d>& vertices() const
  {
    assert(dynamic_cast<mesh_vertex_array<d>*>(verts_.get()));
    return static_cast<const mesh_vertex_array<d>&>(*verts_);
  }
  template <unsigned int d>
  mesh_vertex_array<d>& vertices()
  {
    assert(dynamic_cast<mesh_vertex_array<d>*>(verts_.get()));
    return static_cast<mesh_vertex_array<d>&>(*verts_);
  }

  /// Access the vector of faces
  const mesh_face_array_base& faces() const { return *faces_; }
  mesh_face_array_base& faces() { return *faces_; }

  /// Set the vertices
  void set_vertices(std::unique_ptr<mesh_vertex_array_base> verts) { verts_ = std::move(verts); }

  /// Set the faces
  void set_faces(std::unique_ptr<mesh_face_array_base> faces) { faces_ = std::move(faces); }

  /// Returns true if the mesh has computed half edges
  bool has_half_edges() const { return half_edges_.size() > 0; }

  /// Return the half edge set
  const mesh_half_edge_set& half_edges() const { return half_edges_; }

  /// Construct the half edges graph structure
  void build_edge_graph();

  /// Remove the half edge graph structure
  void remove_edge_graph() { half_edges_.clear(); }

  /// Compute vertex normals
  void compute_vertex_normals();

  /// Compute vertex normals using face normals
  void compute_vertex_normals_from_faces();

  /// Compute face normals
  /**
   * If norm == false the vector lengths are twice the area of the face
   */
  void compute_face_normals(bool norm = true);

  /// This type indicates how texture coordinates are indexed
  /**
   * ON_VERT is one coordinate per vertex
   * ON_CORNER is one coordinate per half edge (i.e. corner)
   */
  enum tex_coord_type { TEX_COORD_NONE = 0,
                        TEX_COORD_ON_VERT = 1,
                        TEX_COORD_ON_CORNER = 2 };

  /// Returns texture coordinate availability
  tex_coord_type has_tex_coords() const { return tex_coord_status_; }

  /// Return the texture coordinates
  const std::vector<vector_2d>& tex_coords() const { return tex_coords_; }

  /// Set the texture coordinates
  void set_tex_coords(const std::vector<vector_2d>& tc);

  /// set the texture sources
  void set_tex_source(const std::string ts) { tex_source_ = ts; }
  const std::string& tex_source() const { return tex_source_; }

  /// Return a vector indicating which faces have texture
  const std::vector<bool>& valid_tex_faces() const { return valid_tex_faces_; }

  /// Set the vector indicating which faces have texture
  void set_valid_tex_faces(const std::vector<bool>& valid);

  /// Label all faces with positive (counter clockwise orientation) area as valid
  /**
   * This requirement refers to the texture map coordinates
   */
  void label_ccw_tex_faces_valid();

  /// Map a barycentric coordinate (u,v) on triangle \param tri into texture space
  vector_2d texture_map(unsigned int tri,
                                double u, double v) const;


private:
  /// array of mesh vertices
  std::unique_ptr<mesh_vertex_array_base> verts_;
  /// array of mesh faces
  std::unique_ptr<mesh_face_array_base> faces_;
  /// set of mesh half edges
  mesh_half_edge_set half_edges_;

  /// vector of texture coordinates
  std::vector<vector_2d> tex_coords_;

  /// vector of texture sources
  std::string tex_source_;

  /// indicate which faces have texture data
  std::vector<bool> valid_tex_faces_;
  /// the type of texture coordinates
  tex_coord_type tex_coord_status_;
};

//shared pointer typedef
typedef std::shared_ptr<mesh> mesh_sptr;



} // end namespace vital
} // end namespace kwiver

#endif // VITAL_MESH_H_
