section .data
    global tensor
    tensor db 10 dup(0)

section .text
    global init_tensor, shift_memory, protect_data, access_data

; Initialize tensor to zero
init_tensor:
    mov edi, tensor
    mov ecx, 10
    xor eax, eax
.init_loop:
    stosb
    loop .init_loop
    ret

; Shift memory by storing new data at a specified index
shift_memory:
    mov esi, [esp+4]  ; Load index
    mov al, [esp+8]   ; Load data
    mov edi, tensor
    add edi, esi
    mov [edi], al
    ret

; Protect data by masking it at a specified index
protect_data:
    mov esi, [esp+4]  ; Load index
    mov edi, tensor
    add edi, esi
    mov byte [edi], 0xFF
    ret

; Access data quickly from the specified index
access_data:
    mov esi, [esp+4]  ; Load index
    mov edi, tensor
    add edi, esi
    mov al, [edi]
    ret
