## Iris Imager

Iris is a ray-tracer built alongside KHARMA, using Kokkos and Parthenon.  Currently it is only useful for imaging KHARMA files: it can produce unpolarized and polarized images at radio frequencies comparable to e.g. Event Horizon Telescope observations, and performs within expected variance compared to other polarized ray-tracers in the polarized comparison problems (i.e. the thin disk and sample dump).

However, it is written to be adapted to many other cases, e.g. automatic flux fitting, multiple cameras and frequency bundling, and eventually Monte Carlo spectra.

Iris owes its existence to/uses a lot of code from/shamelessly rips off [ipole](https://github.com/AFD-Illinois/ipole).  A few 

Code from `ipole` is used under the BSD 3-clause license, and Iris is provided with KHARMA under the same license.  I also copied the ipole log/text output format, so anything relying on that for parsing should work fine.
