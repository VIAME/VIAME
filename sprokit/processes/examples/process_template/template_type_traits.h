// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

//++ rename include guard based on the applicatiopn
#ifndef TEMPLATE_TYPE_TRAITS_H
#define TEMPLATE_TYPE_TRAITS_H

#include <sprokit/process/trait_utils.h>

/*
 * A type trait is used to bind a local trait name to a application wide
 * canonical type name string to a c++ type name. This is useful way
 * to establish names for types that are used throughout a sprokit
 * pipeline.
 *
 * Type traits should name a logical or semantic type not a physical
 * or logical type. This essential for verifying the semantics of a
 * pipeline. For example, GSD is usually a double but the trait name
 * should be \b gsd with a type double. It is a really bad idea to
 * name type traits based on the concrete builtin fundamental type
 * such as double or int.
 *
 * The canonical type name is a string that will be used to identify
 * the type and is used to verify the compatibility of process ports
 * when making connections. Only ports with the same canonical type
 * name can be connected.
 *
 * For small systems, these names can specify the logical data item
 * passing through the ports. For larger systems, it may make sense to
 * establish a hierarchical name space. One way to do this is to
 * separate the levels with a ':' character as shown in the
 * examples. Using qualified names reduces the chance of name
 * collisions when two subsystems pick the same logical name for
 * different underlying types.
 */

// ================================================================
//
// Create type traits for common pipeline types.
// These are types that are passed through the pipeline.
//        ( type-trait-name, "canonical_type_name", concrete-type )
//
create_type_trait( frame_number, "template:frame-number", int );

// ================================================================
//
// Create port traits for common port types.  These trait names are
// used in declaring ports in a process and moving data to/from the
// port.
//
//                  ( port-name, type-trait-name, "port-description" )
//
create_port_trait( video_frame_number, frame_number, "Sequential number of video frame." );

#endif /* TEMPLATE_TYPE_TRAITS_H */
