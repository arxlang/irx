; ModuleID = 'main.arx'
source_filename = "main.arx"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local i32 @main() {
  %1 = alloca i32, align 4         ; a
  %2 = alloca i32, align 4         ; b
  store i32 5, i32* %1, align 4    ; a = 5
  store i32 10, i32* %2, align 4   ; b = 10

  %3 = load i32, i32* %1, align 4  ; load a
  %4 = add nsw i32 %3, 1           ; ++a
  store i32 %4, i32* %1, align 4

  %5 = load i32, i32* %2, align 4  ; load b
  %6 = sub nsw i32 %5, 1           ; --b
  store i32 %6, i32* %2, align 4

  %7 = add nsw i32 %4, %6          ; ++a + --b
  ret i32 %7
}
