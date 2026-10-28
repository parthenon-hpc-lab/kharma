#pragma once

template<typename CRTP>
class CoordinatesBase
{
  public:
    Real get_a() const { return 0; }
    bool is_spherical() const { return false; }
    bool is_ks() const { return false; }
}
