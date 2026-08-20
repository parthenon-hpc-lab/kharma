// This file was generated in part with generative AI

#include <Kokkos_Core.hpp>
#include <catch2/catch_test_macros.hpp>

#include "decs.hpp"

TEST_CASE("Kokkos is initialized for KHARMA unit tests", "[kokkos]")
{
    REQUIRE(Kokkos::is_initialized());
    REQUIRE(GR_DIM == 4);
}

TEST_CASE("Kokkos parallel reductions run in unit tests", "[kokkos]")
{
    Kokkos::View<int*> values("values", 4);

    Kokkos::parallel_for("fill_smoke_view", 4, KOKKOS_LAMBDA(const int i)
        {
            values(i) = i + 1;
        });

    int sum = 0;
    Kokkos::parallel_reduce(
        "sum_smoke_view", 4,
        KOKKOS_LAMBDA(const int i, int& local_sum)
        {
            local_sum += values(i);
        },
        sum);
    Kokkos::fence();

    REQUIRE(sum == 10);
}
