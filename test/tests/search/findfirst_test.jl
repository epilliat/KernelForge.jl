@testset "findfirst" begin

    # ── Scalar element types ─────────────────────────────────────────────

    @testset "Int32 – match at first position" begin
        src = AT(Int32[5, 1, 2, 3])
        idx = KF.findfirst(x -> x == 5, src)
        @test idx == 1
    end

    @testset "Int32 – match at last position" begin
        src = AT(Int32[1, 2, 3, 5])
        idx = KF.findfirst(x -> x == 5, src)
        @test idx == 4
    end

    @testset "Int32 – no match returns nothing" begin
        src = AT(Int32[1, 2, 3, 4])
        idx = KF.findfirst(x -> x == 99, src)
        @test isnothing(idx)
    end

    @testset "Int32 – empty array returns nothing" begin
        src = AT(Int32[])
        idx = KF.findfirst(x -> x > 0, src)
        @test isnothing(idx)
    end

    @testset "Int32 – single element, match" begin
        src = AT(Int32[42])
        idx = KF.findfirst(x -> x == 42, src)
        @test idx == 1
    end

    @testset "Int32 – single element, no match" begin
        src = AT(Int32[42])
        idx = KF.findfirst(x -> x == 0, src)
        @test isnothing(idx)
    end

    @testset "Int32 – all elements match, returns first" begin
        src = AT(ones(Int32, 64))
        idx = KF.findfirst(x -> x == 1, src)
        @test idx == 1
    end

    @testset "Float32 – NaN not equal to itself" begin
        src = AT(Float32[1f0, NaN32, 3f0])
        idx = KF.findfirst(x -> isnan(x), src)
        @test idx == 2
    end

    @testset "Float32 – negative threshold" begin
        src = AT(Float32[-3f0, -1f0, 0f0, 2f0])
        idx = KF.findfirst(x -> x >= 0f0, src)
        @test idx == 3
    end

    # ── Sizes around workgroup boundaries ───────────────────────────────

    @testset "size = workgroup - 1, match at end" begin
        n = 31
        data = fill(Int32(0), n)
        data[end] = Int32(1)
        src = AT(data)
        idx = KF.findfirst(x -> x == 1, src)
        @test idx == n
    end

    @testset "size = workgroup, match in middle" begin
        n = 32
        data = fill(Int32(0), n)
        data[16] = Int32(7)
        src = AT(data)
        idx = KF.findfirst(x -> x == 7, src)
        @test idx == 16
    end

    @testset "size = workgroup + 1, match at boundary" begin
        n = 33
        data = fill(Int32(0), n)
        data[33] = Int32(9)
        src = AT(data)
        idx = KF.findfirst(x -> x == 9, src)
        @test idx == 33
    end

    @testset "size spans multiple blocks" begin
        n = 1024
        data = fill(Int32(-1), n)
        data[700] = Int32(0)
        src = AT(data)
        idx = KF.findfirst(x -> x == 0, src)
        @test idx == 700
    end

    # ── 2D arrays ────────────────────────────────────────────────────────

    @testset "2D Int32 matrix – match found (linear index)" begin
        A = AT(reshape(Int32.(1:12), 3, 4))  # column-major
        idx = KF.findfirst(x -> x == 7, A)
        @test idx == CartesianIndex(1, 3)
    end

    @testset "2D matrix – no match" begin
        A = AT(reshape(Int32.(1:12), 3, 4))
        idx = KF.findfirst(x -> x == 99, A)
        @test isnothing(idx)
    end

    # ── Structured type Vec3 ─────────────────────────────────────────────

    @testset "Vec3 – match on first component" begin
        src = AT([Vec3(1f0, 0f0, 0f0), Vec3(2f0, 0f0, 0f0), Vec3(3f0, 0f0, 0f0)])
        idx = KF.findfirst(v -> v.x > 1.5f0, src)
        @test idx == 2
    end

    @testset "Vec3 – match on norm condition" begin
        src = AT([Vec3(0f0, 0f0, 0f0), Vec3(1f0, 1f0, 1f0), Vec3(2f0, 2f0, 2f0)])
        idx = KF.findfirst(v -> v.x^2 + v.y^2 + v.z^2 > 2f0, src)
        @test idx == 2  # norm² of (1,1,1) = 3 > 2
    end

    @testset "Vec3 – no match" begin
        src = AT([Vec3(0f0, 0f0, 0f0), Vec3(0f0, 1f0, 0f0)])
        idx = KF.findfirst(v -> v.x > 10f0, src)
        @test isnothing(idx)
    end

    @testset "Vec3 – match at first position" begin
        src = AT([Vec3(5f0, 0f0, 0f0), Vec3(0f0, 0f0, 0f0), Vec3(0f0, 0f0, 0f0)])
        idx = KF.findfirst(v -> v.x == 5f0, src)
        @test idx == 1
    end

    @testset "Vec3 – multiple matches, returns first" begin
        src = AT([Vec3(1f0, 0f0, 0f0), Vec3(2f0, 0f0, 0f0), Vec3(3f0, 0f0, 0f0)])
        idx = KF.findfirst(v -> v.x > 0f0, src)
        @test idx == 1
    end

    # ── Bool array ───────────────────────────────────────────────────────

    @testset "Bool array – findfirst true" begin
        src = AT(Bool[false, false, true, false])
        idx = KF.findfirst(x -> x, src)
        @test idx == 3
    end

    @testset "Bool array – all false" begin
        src = AT(Bool[false, false, false])
        idx = KF.findfirst(x -> x, src)
        @test isnothing(idx)
    end

    @testset "Nitem" begin
        src = AT(Int32[0, 0, 1])
        idx = KF.findfirst(x -> x == 1, src; Nitem=2)
        @test idx == 3
    end
end