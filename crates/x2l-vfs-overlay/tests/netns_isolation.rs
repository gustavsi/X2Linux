//! Integration test verifying kernel-enforced `CLONE_NEWNET` offline isolation.
//!
//! Runs inside a fresh single-threaded child process (invoking this test binary with
//! `X2L_NETNS_CHILD=1`) so the parent test runner's network namespace is never mutated.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, UdpSocket};
use std::os::unix::fs::MetadataExt;
use std::process::Command;
use std::time::Duration;
use x2l_vfs_overlay::enter_isolated_netns;

fn netns_inode() -> u64 {
    std::fs::metadata("/proc/self/ns/net")
        .expect("stat /proc/self/ns/net")
        .ino()
}

fn is_netns_blocked_err(err: &std::io::Error) -> bool {
    matches!(
        err.raw_os_error(),
        Some(nix::libc::ENETUNREACH)
            | Some(nix::libc::EHOSTUNREACH)
            | Some(nix::libc::EACCES)
            | Some(nix::libc::EPERM)
            | Some(nix::libc::EADDRNOTAVAIL)
    )
}

fn run_isolated_child_probe() {
    let parent_ino: u64 = std::env::var("X2L_NETNS_PARENT_INO")
        .expect("X2L_NETNS_PARENT_INO")
        .parse()
        .expect("parse parent inode");

    match enter_isolated_netns(false) {
        Ok(()) => {}
        Err(err) => {
            // Gracefully skip live namespace probe on restricted AppArmor CI kernels.
            eprintln!("[SKIP] unprivileged CLONE_NEWNET unavailable on this host: {err}");
            return;
        }
    }

    let child_ino = netns_inode();
    assert_ne!(
        child_ino, parent_ino,
        "child process must have a distinct /proc/self/ns/net inode"
    );

    // 1. Outbound IPv4 TCP & UDP connect to TEST-NET-1 (192.0.2.1:9) must fail immediately.
    let v4_dest: SocketAddr = "192.0.2.1:9".parse().unwrap();
    let tcp_v4 = TcpStream::connect_timeout(&v4_dest, Duration::from_millis(250))
        .expect_err("outbound IPv4 TCP must fail in CLONE_NEWNET");
    assert!(
        is_netns_blocked_err(&tcp_v4),
        "expected ENETUNREACH/EACCES for IPv4 TCP, got: {tcp_v4:?}"
    );

    let udp_v4 = UdpSocket::bind("0.0.0.0:0").expect("bind UDP v4");
    let udp_v4_err = udp_v4
        .connect(v4_dest)
        .expect_err("outbound IPv4 UDP connect must fail in CLONE_NEWNET");
    assert!(
        is_netns_blocked_err(&udp_v4_err),
        "expected ENETUNREACH/EACCES for IPv4 UDP, got: {udp_v4_err:?}"
    );

    // 2. Outbound IPv6 UDP connect to documentation prefix [2001:db8::1]:9 must fail immediately.
    if let Ok(udp_v6) = UdpSocket::bind("[::]:0") {
        let v6_dest: SocketAddr = "[2001:db8::1]:9".parse().unwrap();
        let udp_v6_err = udp_v6
            .connect(v6_dest)
            .expect_err("outbound IPv6 UDP connect must fail in CLONE_NEWNET");
        assert!(
            is_netns_blocked_err(&udp_v6_err),
            "expected ENETUNREACH/EACCES for IPv6 UDP, got: {udp_v6_err:?}"
        );
    }

    // 3. Local loopback (127.0.0.1) TCP must remain functional inside the isolated namespace.
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind 127.0.0.1 loopback listener");
    let addr = listener.local_addr().unwrap();
    let mut client = TcpStream::connect(addr).expect("connect to 127.0.0.1 inside netns");
    let (mut server, _) = listener.accept().expect("accept loopback connection");
    client.write_all(b"x2l-lo").unwrap();
    let mut buf = [0u8; 6];
    server.read_exact(&mut buf).unwrap();
    assert_eq!(&buf, b"x2l-lo");
}

#[test]
fn isolated_netns_blocks_outbound_connect_while_keeping_loopback() {
    if std::env::var("X2L_NETNS_CHILD").ok().as_deref() == Some("1") {
        run_isolated_child_probe();
        return;
    }

    let parent_ino_before = netns_inode();
    let exe = std::env::current_exe().expect("current test binary");
    let output = Command::new(exe)
        .arg("--exact")
        .arg("isolated_netns_blocks_outbound_connect_while_keeping_loopback")
        .arg("--nocapture")
        .env("X2L_NETNS_CHILD", "1")
        .env("X2L_NETNS_PARENT_INO", parent_ino_before.to_string())
        .output()
        .expect("spawn netns child probe");

    if !output.status.success() {
        panic!(
            "netns child probe failed ({}):\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    let parent_ino_after = netns_inode();
    assert_eq!(
        parent_ino_before, parent_ino_after,
        "parent network namespace must not change"
    );
}
