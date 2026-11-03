// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

import CoreBluetooth
import Foundation

/// CoreBluetooth central for the `rlx-voice-link-nrf` firmware: finds the
/// device by service UUID, subscribes to audio notifications and hands each
/// complete utterance to `onUtterance`. `startTalking()` / `stopTalking()`
/// are push-to-talk from the phone; the device button works too.
///
/// iOS: add `NSBluetoothAlwaysUsageDescription` to Info.plist, and the
/// `bluetooth-central` background mode to keep streaming in the background.
public final class VoiceLinkCentral: NSObject, CBCentralManagerDelegate, CBPeripheralDelegate {
    public static let service = CBUUID(string: "7A1E0001-5C2B-4E8A-9F43-72C0B1D6A5E3")
    public static let audio = CBUUID(string: "7A1E0002-5C2B-4E8A-9F43-72C0B1D6A5E3")
    public static let control = CBUUID(string: "7A1E0003-5C2B-4E8A-9F43-72C0B1D6A5E3")
    static let cmdStart: UInt8 = 1
    static let cmdStop: UInt8 = 2

    /// A completed utterance, 16 kHz mono PCM, on the main queue.
    public var onUtterance: (([Int16]) -> Void)?
    /// Connection state changes, for UI.
    public var onState: ((String) -> Void)?

    private var central: CBCentralManager!
    private var peripheral: CBPeripheral?
    private var controlChar: CBCharacteristic?
    private let receiver = UtteranceReceiver()

    public override init() {
        super.init()
        central = CBCentralManager(delegate: self, queue: .main)
    }

    /// Packets lost so far on this link.
    public var lostPackets: UInt32 { receiver.lost }

    public func startTalking() { send(Self.cmdStart) }
    public func stopTalking() { send(Self.cmdStop) }

    private func send(_ cmd: UInt8) {
        guard let p = peripheral, let c = controlChar else { return }
        p.writeValue(Data([cmd]), for: c, type: .withResponse)
    }

    public func centralManagerDidUpdateState(_ central: CBCentralManager) {
        guard central.state == .poweredOn else {
            onState?("bluetooth \(central.state.rawValue)")
            return
        }
        onState?("scanning")
        central.scanForPeripherals(withServices: [Self.service])
    }

    public func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral,
                               advertisementData: [String: Any], rssi RSSI: NSNumber) {
        central.stopScan()
        self.peripheral = peripheral
        peripheral.delegate = self
        onState?("connecting \(peripheral.name ?? "rlx-voice")")
        central.connect(peripheral)
    }

    public func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        onState?("connected")
        peripheral.discoverServices([Self.service])
    }

    public func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral,
                               error: Error?) {
        onState?("disconnected; scanning")
        self.peripheral = nil
        controlChar = nil
        central.scanForPeripherals(withServices: [Self.service])
    }

    public func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        for s in peripheral.services ?? [] where s.uuid == Self.service {
            peripheral.discoverCharacteristics([Self.audio, Self.control], for: s)
        }
    }

    public func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService,
                           error: Error?) {
        for c in service.characteristics ?? [] {
            if c.uuid == Self.audio { peripheral.setNotifyValue(true, for: c) }
            if c.uuid == Self.control { controlChar = c }
        }
        let mtu = peripheral.maximumWriteValueLength(for: .withoutResponse) + 3
        onState?("ready (ATT MTU \(mtu))")
    }

    public func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic,
                           error: Error?) {
        guard characteristic.uuid == Self.audio, let v = characteristic.value else { return }
        if let pcm = receiver.push(v) { onUtterance?(pcm) }
    }
}
