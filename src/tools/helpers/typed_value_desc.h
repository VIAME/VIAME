/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_TOOLS_HELPERS_TYPED_VALUE_DESC_H
#define VISTK_TOOLS_HELPERS_TYPED_VALUE_DESC_H

#include <boost/program_options/value_semantic.hpp>
#include <boost/any.hpp>

/*
 * This file may be removed once Boost bug #4781
 * (<https://svn.boost.org/trac/boost/ticket/4781>) is resolved.
 */

namespace boost
{

namespace program_options
{

template <typename T, typename charT = char>
class typed_value_desc
  : public typed_value<T, charT>
{
  public:
    typedef typed_value<T, charT> base_t;

    typed_value_desc(T* v)
      : base_t(v)
    {
    }

    std::string name() const
    {
      if (!m_metavar.empty())
      {
        if (!m_implicit_value.empty() && !m_implicit_value_as_text.empty())
        {
            std::string msg = "[=" + m_metavar + "(=" + m_implicit_value_as_text + ")]";
            if (!m_default_value.empty() && !m_default_value_as_text.empty())
                msg += " (=" + m_default_value_as_text + ")";
            return msg;
        }
        else if (!m_default_value.empty() && !m_default_value_as_text.empty())
        {
            return m_metavar + " (=" + m_default_value_as_text + ")";
        }
        else
        {
            return m_metavar;
        }
      }
      else
      {
        return base_t::name();
      }
    }

    typed_value_desc* default_value(const T& v)
    {
      m_default_value = boost::any(v);
      m_default_value_as_text = boost::lexical_cast<std::string>(v);
      base_t::default_value(v);
      return this;
    }

    typed_value_desc* default_value(const T& v, const std::string& textual)
    {
      m_default_value = boost::any(v);
      m_default_value_as_text = textual;
      base_t::default_value(v, textual);
      return this;
    }

    typed_value_desc* implicit_value(const T &v)
    {
      m_implicit_value = boost::any(v);
      m_implicit_value_as_text = boost::lexical_cast<std::string>(v);
      base_t::implicit_value(v);
      return this;
    }

    typed_value_desc* implicit_value(const T &v, const std::string& textual)
    {
      m_implicit_value = boost::any(v);
      m_implicit_value_as_text = textual;
      base_t::implicit_value(v, textual);
      return this;
    }

    typed_value_desc* metavar(std::string const& meta)
    {
      m_metavar = meta;
      return this;
    }
  protected:
    boost::any m_default_value;
    std::string m_default_value_as_text;
    boost::any m_implicit_value;
    std::string m_implicit_value_as_text;

    std::string m_metavar;
};

template <typename T>
typed_value_desc<T>*
value_desc()
{
  return new typed_value_desc<T>(NULL);
}

template <typename T>
typed_value_desc<T>*
value_desc(T* v)
{
  typed_value_desc<T>* const r = new typed_value_desc<T>(v);

  return r;
}

template <typename T>
typed_value_desc<T, wchar_t>*
wvalue_desc()
{
  return new typed_value_desc<T, wchar_t>(NULL);
}

template <typename T>
typed_value_desc<T, wchar_t>*
wvalue_desc(T* v)
{
  typed_value_desc<T, wchar_t>* const r = new typed_value_desc<T, wchar_t>(v);

  return r;
}

}

}

#endif // VISTK_TOOLS_HELPERS_TYPED_VALUE_DESC_H
