# Cross-backend compile-only hook used by the autotune script (see
# data/tuning/matvec/autotune.jl). Each backend extension implements this
# on its concrete `KA.Kernel{...}` type, mirroring the KA functor's prelude
# but stopping at the `@cuda launch=false` / `@roc launch=false` step so
# only the in-process GPUCompiler cache is populated — no module load,
# kernel execute, or sync.
#
# With no backend extension loaded this is undefined; the autotune is the
# only caller and always runs with a backend loaded.
function compile_kernel_only end
