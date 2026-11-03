/* nRF54H20 application core, mapped onto QEMU's MPS2 AN505 (Cortex-M33).
   Sizes are the nRF54H20's, so an image that links here fits the chip:
     MRAM  2 MB   code and constants (MRAM_10 + MRAM_11)
     RAM   768 KB scratch, stack, .data (general-purpose global RAM, RAM0x,
                  if the application core owns all of it)
     EXT   16 MB  the model, read in place from external flash on EXMIF
   On QEMU the three live in SSRAM1, SSRAM2/3 and PSRAM (secure aliases).
   On silicon only the ORIGINs change. */
MEMORY {
  MRAM : ORIGIN = 0x10000000, LENGTH = 2M
  RAM  : ORIGIN = 0x38000000, LENGTH = 768K
  EXT  : ORIGIN = 0x80000000, LENGTH = 16M
}
ENTRY(reset)
SECTIONS {
  .vectors : { KEEP(*(.vectors)) } > MRAM
  .text    : { *(.text .text.*) } > MRAM
  .rodata  : { *(.rodata .rodata.*) } > MRAM
  .exmif : { KEEP(*(.exmif)) } > EXT
  .data : {
    . = ALIGN(4); __data_start = .;
    *(.data .data.*)
    . = ALIGN(4); __data_end = .;
  } > RAM AT > MRAM
  __data_load = LOADADDR(.data);
  .bss (NOLOAD) : {
    . = ALIGN(4); __bss_start = .;
    *(.bss .bss.*) *(COMMON)
    . = ALIGN(4); __bss_end = .;
  } > RAM
  /* Stack: what is left of RAM, at least 16 KB. */
  _stack_top = ORIGIN(RAM) + LENGTH(RAM);
  ASSERT(_stack_top - __bss_end >= 16K, "RAM: under 16 KB left for the stack")
  /DISCARD/ : { *(.ARM.exidx .ARM.exidx.* .ARM.extab.*) }
}
