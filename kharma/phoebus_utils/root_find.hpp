// © 2021. Triad National Security, LLC. All rights reserved.  This
// program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
// is operated by Triad National Security, LLC for the U.S.
// Department of Energy/National Nuclear Security Administration. All
// rights in the program are reserved by Triad National Security, LLC,
// and the U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works,
// distribute copies to the public, perform publicly and display
// publicly, and to permit others to do so.

#ifndef PHOEBUS_UTILS_ROOT_FIND_HPP_
#define PHOEBUS_UTILS_ROOT_FIND_HPP_

namespace root_find {

enum class RootFindStatus { success, failure };

struct RootFind {
  KOKKOS_INLINE_FUNCTION
  RootFind(int max_iterations = std::numeric_limits<int>::max())
      : iteration_count(0), max_iter(max_iterations) {}

  KOKKOS_INLINE_FUNCTION
  bool check_bracket(const Real a, const Real b, const Real ya, const Real yb,
                     RootFindStatus *status = nullptr) {
    if (ya * yb > 0.0) {
      if (status != nullptr) {
        *status = RootFindStatus::failure;
      } else {
        printf("Root not bracketed in find_root. a, b, ya, yb = %g %g %g %g\n", a, b, ya,
               yb);
      }
      return false;
    }
    return true;
  }

  template <typename F>
  KOKKOS_INLINE_FUNCTION void refine_bracket(F &func, const Real guess, Real &a, Real &b,
                                             Real &ya, Real &yb) {
    if (a <= guess && b >= guess) {
      ya = func(guess);
      yb = func(b);
      if (ya * yb > 0.0) {
        yb = ya;
        b = guess;
        ya = func(a);
      } else {
        a = guess;
      }
    } else {
      ya = func(a);
      yb = func(b);
    }
  }

  template <typename F>
  KOKKOS_INLINE_FUNCTION Real secant(F &func, Real a, Real b, const Real tol,
                                     const Real guess, RootFindStatus *status = nullptr) {
    Real ya, yb;
    refine_bracket(func, guess, a, b, ya, yb);
    Real x1, x2, y1, y2;
    if (a == guess) {
      x1 = a;
      y1 = ya;
      x2 = b;
      y2 = yb;
    } else if (b == guess) {
      x1 = b;
      y1 = yb;
      x2 = a;
      y2 = ya;
    }
    Real sign = (y1 < 0 ? 1.0 : -1.0);
    y1 *= sign;
    y2 *= sign;

    while (std::abs(x2 - x1) > 2.0 * tol && iteration_count < max_iter) {
      Real x0 = (x1 * y2 - x2 * y1) / (y2 - y1);
      // guard against roundoff because ya or yb is sufficiently close to zero
      if (x0 == x1) {
        x2 = x1;
        iteration_count++;
        continue;
      } else if (x0 == x2) {
        x1 = x2;
        iteration_count++;
        continue;
      }
      Real y0 = sign * func(x0);
      x2 = x1;
      y2 = y1;
      x1 = x0;
      y1 = y0;
      iteration_count++;
    }

    if (status != nullptr) {
      *status =
          iteration_count == max_iter ? RootFindStatus::failure : RootFindStatus::success;
    }

    return 0.5 * (x1 + x2);
  }

  template <typename F>
  KOKKOS_INLINE_FUNCTION Real regula_falsi(F &func, Real a, Real b, const Real tol,
                                           const Real guess,
                                           RootFindStatus *status = nullptr) {
    Real ya, yb;
    refine_bracket(func, guess, a, b, ya, yb);
    if (!check_bracket(a, b, ya, yb, status)) {
      if (status == nullptr) {
        PARTHENON_FAIL("Aborting with unbracketed root.");
      } else {
        *status = RootFindStatus::failure;
        return guess;
      }
    }
    Real sign = (ya < 0 ? 1.0 : -1.0);
    ya *= sign;
    yb *= sign;

    int b1 = 0;
    int b2 = 0;
    while (b - a > 2.0 * tol && iteration_count < max_iter) {
      Real c = (a * yb - b * ya) / (yb - ya);
      // guard against roundoff because ya or yb is sufficiently close to zero
      if (c == a) {
        b = a;
        iteration_count++;
        continue;
      } else if (c == b) {
        a = b;
        iteration_count++;
        continue;
      }
      Real yc = sign * func(c);
      if (yc > 0.0) {
        b = c;
        yb = yc;
        b1++;
        ya *= (b1 > 1 ? 0.5 : 1.0);
        b2 = 0;
      } else if (yc < 0.0) {
        a = c;
        ya = yc;
        b2++;
        yb *= (b2 > 1 ? 0.5 : 1.0);
        b1 = 0;
      } else {
        a = c;
        b = c;
      }
      iteration_count++;
    }

    if (status != nullptr) {
      *status =
          iteration_count == max_iter ? RootFindStatus::failure : RootFindStatus::success;
    }

    return 0.5 * (a + b);
  }

  template <typename F>
  KOKKOS_INLINE_FUNCTION Real itp(F &func, Real a, Real b, const Real tol,
                                  const Real guess, RootFindStatus *status = nullptr) {
    constexpr Real kappa1 = 0.1;
    constexpr Real kappa2 = 2.0;
    constexpr int n0 = 0;

    const int nmax = std::ceil(std::log2(0.5 * (b - a) / tol)) + n0;

    Real ya, yb;
    refine_bracket(func, guess, a, b, ya, yb);
    if (!check_bracket(a, b, ya, yb, status)) {
      if (status == nullptr) {
        PARTHENON_FAIL("Aborting with unbracketed root.");
      } else {
        *status = RootFindStatus::failure;
        return guess;
      }
    }
    Real sign = (ya < 0 ? 1.0 : -1.0);
    ya *= sign;
    yb *= sign;

    while (b - a > 2.0 * tol && iteration_count < max_iter) {
      const Real xh = 0.5 * (a + b);
      const Real xf = (yb * a - ya * b) / (yb - ya);
      const Real xhxf = xh - xf;
      const Real delta = std::min(kappa1 * std::pow(b - a, kappa2), std::abs(xhxf));
      const Real sigma = (xhxf > 0 ? 1.0 : -1.0);
      const Real xt = (delta <= sigma * xhxf ? xf + sigma * delta : xh);
      const Real r = tol * std::pow(2.0, nmax - iteration_count) - 0.5 * (b - a);
      const Real xitp = (std::fabs(xt - xh) > r ? xh - sigma * r : xt);
      const Real yitp = sign * func(xitp);
      if (yitp > 0.0) {
        b = xitp;
        yb = yitp;
      } else if (yitp < 0.0) {
        a = xitp;
        ya = yitp;
      } else {
        a = xitp;
        b = xitp;
      }
      iteration_count++;
    }

    if (status != nullptr) {
      *status =
          iteration_count == max_iter ? RootFindStatus::failure : RootFindStatus::success;
    }

    return 0.5 * (a + b);
  }

  template <typename F>
  KOKKOS_INLINE_FUNCTION Real bisect(F func, Real a, Real b, const Real tol,
                                     RootFindStatus *status = nullptr) {

    Real ya = func(a);
    Real yb = func(b);
    if (!check_bracket(a, b, ya, yb, status)) {
      if (status == nullptr) {
        PARTHENON_FAIL("Aborting with unbracketed root.");
      } else {
        *status = RootFindStatus::failure;
        return (a + b) / 2.;
      }
    }
    Real sign = (ya < 0 ? 1.0 : -1.0);
    ya *= sign;
    yb *= sign;

    while (b - a > tol && iteration_count < max_iter) {
      const Real xh = 0.5 * (a + b);
      const Real yh = sign * func(xh);
      if (yh > 0.0) {
        b = xh;
        yb = yh;
      } else if (yh < 0.0) {
        a = xh;
        ya = yh;
      } else {
        a = xh;
        b = xh;
      }
      iteration_count++;
    }

    if (status != nullptr) {
      *status =
          iteration_count == max_iter ? RootFindStatus::failure : RootFindStatus::success;
    }

    return 0.5 * (a + b);
  }

  int iteration_count, max_iter;
};
} // namespace root_find

#endif // PHOEBUS_UTILS_ROOT_FIND_HPP_
