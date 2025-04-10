; ModuleID = 'if_stmt'
source_filename = "if_stmt"

define dso_local i32 @main() {
entry:
  ; int a = 10;
  %a = alloca i32
  store i32 10, i32* %a

  ; if (a > 5)
  %a_val = load i32, i32* %a
  %cmp = icmp sgt i32 %a_val, 5
  br i1 %cmp, label %then, label %else

then:
  ; then block returns 1
  %then_val = add i32 1, 0 ; just yield constant 1
  br label %ifcont

else:
  ; else block returns 0
  %else_val = add i32 0, 0 ; just yield constant 0
  br label %ifcont

ifcont:
  ; phi node merges then and else
  %iftmp = phi i32 [ %then_val, %then ], [ %else_val, %else ]
  ret i32 %iftmp
}
