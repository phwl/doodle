# LQG Setup

This guide describes the prerequisites and setup steps required to run the LQG controller.

## Prerequisites

- RedPitaya STEMlab 125-14 Z7020 (Gen 1 or 2), with OS 2.07-43 or newer
- MATLAB 2024a or newer 
- Compatible FPGA bitstream - `lqg_fp7_v1.2.2.bin`
- Optional: Matlab toolboxes and addons to run the GUI source code:
   - Control Systems Toolbox
   - Image Processing Toolbox
   - Optimization Toolbox
   - Signal Processing Toolbox
   - Advanced Logger for MATLAB 2.0.2 (https://www.mathworks.com/matlabcentral/fileexchange/87322-advanced-logger-for-matlab)
- Optional: Internet connection for the Red Pitaya
- Optional: Oscilloscope and signal generator for I/O calibration

If you plan to use the LQG data recording feature, reserve memory in the Red Pitaya device tree as described below.

## Red Pitaya Memory Reservation

To support LQG GUI data recording, reserve at least 256 MB in onboard DDR memory.

1. Unmount the SD card from the Red Pitaya (RP) and mount it on a PC.
2. Open `dts/z20_125/dtraw.dts` from the boot partition.
3. Find this block in the `memory` section:

```dts
...
memory {
    device_type = "memory";
    reg = <0x00 0x20000000>;
};
...
```

4. Change `reg` to:

```dts
reg = <0x00 0x10000000>;
```

5. Compile the device tree:

```bash
dtc -O dtb -o devicetree.dtb dtraw.dts
```

6. Save changes, eject the SD card, and put it back into the RP.
7. After reconnecting to the RP, verify memory config:

```bash
head /proc/meminfo
```

## Hardware Setup

1. Ensure the RP is accessible via SSH and SCP (WinSCP and PuTTY are recommended on Windows).
2. Ensure these packages are installed on the RP:
   - `python3-yaml`
   - `python3-bitarray`
   - `python3-numpy`
   - `python3-flask`

   If RP has internet access:

```bash
apt-get install python3-yaml python3-bitarray python3-numpy python3-flask
```

3. Copy the full `firmware/coreconfig` folder (Python configuration server) to the RP via SCP.
4. Copy `ip_description.yaml` to the parent directory of `coreconfig`.
5. Copy the LQG bitstream (for example `firmware/vivado/bitstreams/lqg_fp7_v1.2.2.bin`) to `lib/firmware/`.
6. From the parent directory of `coreconfig`, load the bitstream:

```bash
echo "lqg_fp7_v1.2.2.bin" > /sys/class/fpga_manager/fpga0/firmware
```

7. Start the server:

```bash
./coreconfig/server.py ip_description.yaml
```

## Host PC Setup

1. Clone the `LQG/user-interface` repository to the host PC.
2. Navigate into the `user-interface` folder.
3. Either:
   - Start the compiled application in `/release/`, or
   - Open `LQG_FP7_GUI.mlapp` and launch the GUI from source.

## Next Steps

- For technical GUI documentation, see `LQG_FP7/user-interface/docs` (also available from the GUI under the **User guide** tab).
- If you discover any bugs, reach out to us via the contacts below. Please include the relevant log files and as rich as possible description of the issue.

## Opening Docs From Windows File Explorer

From File Explorer, go to `user-interface/docs` and double-click `OPEN_DOCS.bat`.

- This starts a small local web server and opens the documentation at `http://127.0.0.1:8931/index.html`.
- The launcher uses Python if available; otherwise it automatically uses a native PowerShell server (`serve-docs.ps1`).


## Contacts
- Vojtěch Mlynář - mlynar@acin.tuwien.ac.at
- Johannes Berndorfer - johannes.berndorfer@tuwien.ac.at
- Andreas Deutschmann-Olek - deutschmann@acin.tuwien.ac.at
- https://www.acin.tuwien.ac.at/en/

## License
This user interface for the LQG-FP7, including the FPGA implementation by Vojtěch Mlynář and Johannes Berndorfer is licensed under [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/). The users acknowledge by using this software that the software is provided "as is", with all faults and without warranty of any kind.