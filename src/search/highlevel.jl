argmax(rel, src::AbstractArray) = argmax1d(identity, rel, src)
argmax(src) = argmax(>, src)
argmin(src) = argmax(<, src)


