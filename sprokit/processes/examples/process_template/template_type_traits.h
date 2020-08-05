/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
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
