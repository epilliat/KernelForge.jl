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