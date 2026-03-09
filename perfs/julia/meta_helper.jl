function has_cuda()
    try
        run(pipeline(`nvidia-smi`, devnull))
        return true
    catch
        return false
    end
end

function has_roc()
    try
        run(pipeline(`rocm-smi`, devnull))
        return true
    catch
        return false
    end
end

has_metal() = Sys.isapple()
backend_str = has_roc() ? "roc" :
              has_cuda() ? "cuda" :
              has_metal() ? "metal" :
              error("No supported GPU backend found")