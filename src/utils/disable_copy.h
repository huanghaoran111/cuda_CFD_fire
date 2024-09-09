#pragma once

struct DisableCopyAllowMove
{
    DisableCopyAllowMove() = default;
    DisableCopyAllowMove(DisableCopyAllowMove const&) = delete;
    DisableCopyAllowMove& operator=(DisableCopyAllowMove const&) = delete;
    DisableCopyAllowMove(DisableCopyAllowMove &&) = default;
    DisableCopyAllowMove &operator=(DisableCopyAllowMove &&) = default;
};

struct AllowCopyDisableMove
{
    AllowCopyDisableMove() = default;
    AllowCopyDisableMove(AllowCopyDisableMove const&) = default;
    AllowCopyDisableMove& operator=(AllowCopyDisableMove const&) = default;
    AllowCopyDisableMove(AllowCopyDisableMove &&) = delete;
    AllowCopyDisableMove &operator=(AllowCopyDisableMove &&) = delete;
};

struct DisableCopyDisableMove
{
    DisableCopyDisableMove() = default;
    DisableCopyDisableMove(DisableCopyDisableMove const&) = delete;
    DisableCopyDisableMove& operator=(DisableCopyDisableMove const&) = delete;
    DisableCopyDisableMove(DisableCopyDisableMove &&) = delete;
    DisableCopyDisableMove &operator=(DisableCopyDisableMove &&) = delete;
};
