/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include <gtest/gtest.h>

#include "utilities_training.h"

#include <stdexcept>

TEST( utilities_training, filtered_sequence_partitions )
{
  const std::vector< std::vector< std::string > > items = {
    { "a0", "a1", "a2" }, { "b0", "b1" } };
  const auto train = viame::partition_sequences( items, { "a0", "a2" } );
  const auto validation = viame::partition_sequences( items, { "b0", "b1" } );
  EXPECT_EQ( train.count, ( std::vector< std::size_t >{ 2, 0 } ) );
  EXPECT_EQ( validation.count, ( std::vector< std::size_t >{ 0, 2 } ) );
  EXPECT_EQ( validation.first, ( std::vector< std::size_t >{ 0, 0 } ) );

  const auto filtered = viame::partition_sequences( items, { "b1", "a2" } );
  EXPECT_EQ( filtered.images, ( std::vector< std::string >{ "a2", "b1" } ) );
  EXPECT_EQ( filtered.first, ( std::vector< std::size_t >{ 0, 1 } ) );
}

TEST( utilities_training, rejects_ambiguous_sequence_ownership )
{
  EXPECT_THROW(
    viame::partition_sequences( { { "same" }, { "same" } }, { "same" } ),
    std::runtime_error );
}

TEST( utilities_training, rejects_selected_frame_without_sequence )
{
  EXPECT_THROW(
    viame::partition_sequences( { { "a0" } }, { "missing" } ),
    std::runtime_error );
}
