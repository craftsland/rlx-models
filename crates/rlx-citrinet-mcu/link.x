/* Bare-metal layout for QEMU's rv32 `virt` board with -m 64M. On an ESP32-P4
   the model goes to memory-mapped flash and the scratch to PSRAM; only these
   addresses change. */
MEMORY {
  RAM : ORIGIN = 0x80000000, LENGTH = 64M
}
ENTRY(_start)
SECTIONS {
  .init  : { KEEP(*(.init)) }        > RAM
  .text  : { *(.text .text.*) }      > RAM
  .rodata : { *(.rodata .rodata.*) *(.srodata .srodata.*) } > RAM
  .data  : { *(.data .data.*) *(.sdata .sdata.*) } > RAM
  . = ALIGN(4);
  __bss_start = .;
  .bss (NOLOAD) : { *(.bss .bss.*) *(.sbss .sbss.*) *(COMMON) } > RAM
  . = ALIGN(4);
  __bss_end = .;
  _stack_top = ORIGIN(RAM) + LENGTH(RAM);
  /DISCARD/ : { *(.eh_frame .eh_frame_hdr) }
}
