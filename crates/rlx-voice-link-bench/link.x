/* QEMU MPS2 AN505: code in SSRAM1, data in SSRAM2/3 (secure aliases). */
MEMORY {
  CODE : ORIGIN = 0x10000000, LENGTH = 4M
  RAM  : ORIGIN = 0x38000000, LENGTH = 4M
}
ENTRY(reset)
SECTIONS {
  .vectors : { KEEP(*(.vectors)) } > CODE
  .text    : { *(.text .text.*) } > CODE
  .rodata  : { *(.rodata .rodata.*) } > CODE
  .data : {
    . = ALIGN(4); __data_start = .;
    *(.data .data.*)
    . = ALIGN(4); __data_end = .;
  } > RAM AT > CODE
  __data_load = LOADADDR(.data);
  .bss (NOLOAD) : {
    . = ALIGN(4); __bss_start = .;
    *(.bss .bss.*) *(COMMON)
    . = ALIGN(4); __bss_end = .;
  } > RAM
  _stack_top = ORIGIN(RAM) + LENGTH(RAM);
  /DISCARD/ : { *(.ARM.exidx .ARM.exidx.* .ARM.extab.*) }
}
