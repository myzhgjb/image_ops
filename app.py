import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import cv2 as cv
import numpy as np

from image_ops import (
    read_image, save_image,
    sketch_effect, oil_painting_effect, cartoon_effect,
    color_transfer_lab, histogram_match_rgb,
    pyramid_texture_blend,
    ensure_uint8,
)


class ImageApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title('数字图像风格迁移与特效生成系统')
        self.geometry('1200x720')

        self.content_img = None
        self.style_img = None
        self.preview_img = None

        self._build_ui()

        self.method_var.set('sketch')
        self._update_params_panel()

    def _build_ui(self):
        # 顶部控制区
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Button(top, text='打开内容图', command=self.load_content).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text='打开风格/纹理图', command=self.load_style).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text='保存结果', command=self.save_result).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text='批量处理', command=self.batch_process).pack(side=tk.LEFT, padx=4)

        self.method_var = tk.StringVar()
        ttk.Label(top, text='方法:').pack(side=tk.LEFT, padx=(16, 4))
        method_box = ttk.Combobox(top, textvariable=self.method_var, state='readonly',
                                  values=['sketch', 'oil', 'cartoon', 'color_lab', 'hist_match', 'texture_blend'])
        method_box.pack(side=tk.LEFT)
        method_box.bind('<<ComboboxSelected>>', lambda e: self._update_params_panel())

        ttk.Button(top, text='应用', command=self.apply_effect).pack(side=tk.LEFT, padx=8)

        # 参数面板
        self.params_frame = ttk.LabelFrame(self, text='参数')
        self.params_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=6)

        # 预览区
        self.canvas = tk.Canvas(self, bg='#222')
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 具体参数变量
        self.var_blur = tk.IntVar(value=21)
        self.var_edge_low = tk.IntVar(value=60)
        self.var_edge_high = tk.IntVar(value=150)
        self.var_bilateral_iters = tk.IntVar(value=6)
        self.var_strength = tk.IntVar(value=9)
        self.var_quant = tk.IntVar(value=24)

    def _clear_params(self):
        for w in self.params_frame.winfo_children():
            w.destroy()

    def _update_params_panel(self):
        self._clear_params()
        method = self.method_var.get()
        add = self._add_slider
        if method == 'sketch':
            add('模糊核', self.var_blur, 3, 51, 2)
            add('边缘低阈值', self.var_edge_low, 0, 255, 1)
            add('边缘高阈值', self.var_edge_high, 0, 255, 1)
        elif method == 'oil':
            add('强度/核大小', self.var_strength, 3, 21, 2)
            add('颜色量化级数', self.var_quant, 6, 48, 1)
        elif method == 'cartoon':
            add('双边迭代', self.var_bilateral_iters, 1, 10, 1)
            add('边缘低阈值', self.var_edge_low, 0, 255, 1)
            add('边缘高阈值', self.var_edge_high, 0, 255, 1)
        else:
            ttk.Label(self.params_frame, text='此方法无需额外参数或使用默认参数').pack(anchor='w', padx=6, pady=6)

    def _add_slider(self, text, var, frm, to, step):
        row = ttk.Frame(self.params_frame)
        row.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(row, text=text, width=12).pack(side=tk.LEFT)
        s = ttk.Scale(row, from_=frm, to=to, orient=tk.HORIZONTAL, variable=var, command=lambda v: self._debounced_preview())
        s.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(row, textvariable=var, width=6).pack(side=tk.RIGHT)

    def load_content(self):
        path = filedialog.askopenfilename(filetypes=[('Image', '*.png;*.jpg;*.jpeg;*.bmp')])
        if not path:
            return
        self.content_img = read_image(path)
        self._show_preview(self.content_img)

    def load_style(self):
        path = filedialog.askopenfilename(filetypes=[('Image', '*.png;*.jpg;*.jpeg;*.bmp')])
        if not path:
            return
        self.style_img = read_image(path)
        messagebox.showinfo('提示', '风格/纹理图已加载')

    def save_result(self):
        if self.preview_img is None:
            messagebox.showwarning('提示', '没有结果可以保存')
            return
        path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png'), ('JPEG', '*.jpg')])
        if not path:
            return
        save_image(path, self.preview_img)
        messagebox.showinfo('保存成功', f'已保存到\n{path}')

    def _debounced_preview(self):
        if hasattr(self, '_debounce_timer') and self._debounce_timer is not None:
            self.after_cancel(self._debounce_timer)
        self._debounce_timer = self.after(250, self.apply_effect)

    def apply_effect(self):
        if self.content_img is None:
            return
        method = self.method_var.get()
        img = self.content_img
        try:
            if method == 'sketch':
                res = sketch_effect(img, 'gray', self.var_blur.get(), 0, self.var_edge_low.get(), self.var_edge_high.get())
            elif method == 'oil':
                res = oil_painting_effect(img, 'bilateral', self.var_strength.get(), self.var_quant.get(), True)
            elif method == 'cartoon':
                res = cartoon_effect(img, self.var_bilateral_iters.get(), self.var_edge_low.get(), self.var_edge_high.get())
            elif method == 'color_lab':
                if self.style_img is None:
                    messagebox.showwarning('缺少风格图', '请先加载风格图')
                    return
                res = color_transfer_lab(img, self.style_img)
            elif method == 'hist_match':
                if self.style_img is None:
                    messagebox.showwarning('缺少风格图', '请先加载风格图')
                    return
                res = histogram_match_rgb(img, self.style_img)
            elif method == 'texture_blend':
                if self.style_img is None:
                    messagebox.showwarning('缺少纹理图', '请先加载纹理图')
                    return
                res = pyramid_texture_blend(img, self.style_img, 4)
            else:
                res = img
            self._show_preview(res)
        except Exception as e:
            messagebox.showerror('错误', str(e))

    def _show_preview(self, img_bgr):
        self.preview_img = ensure_uint8(img_bgr)
        # 自适应到画布
        cW = self.canvas.winfo_width() or 800
        cH = self.canvas.winfo_height() or 600
        h, w = self.preview_img.shape[:2]
        scale = min(cW / w, cH / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        disp = cv.resize(self.preview_img, (nw, nh), interpolation=cv.INTER_AREA)
        rgb = cv.cvtColor(disp, cv.COLOR_BGR2RGB)
        from PIL import Image, ImageTk
        im = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=im)
        self.canvas.delete('all')
        self.canvas.create_image(cW // 2, cH // 2, image=imgtk, anchor=tk.CENTER)
        self.canvas.image = imgtk

    def batch_process(self):
        if self.content_img is None:
            messagebox.showwarning('提示', '请先加载任意一张内容图用于确定方法与参数')
            return
        in_dir = filedialog.askdirectory(title='选择输入文件夹')
        if not in_dir:
            return
        out_dir = filedialog.askdirectory(title='选择输出文件夹')
        if not out_dir:
            return
        method = self.method_var.get()
        edge_low = self.var_edge_low.get()
        edge_high = self.var_edge_high.get()
        blur = self.var_blur.get()
        bi_iters = self.var_bilateral_iters.get()
        strength = self.var_strength.get()
        quant = self.var_quant.get()
        style = self.style_img

        def worker():
            for name in os.listdir(in_dir):
                if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    path = os.path.join(in_dir, name)
                    try:
                        img = read_image(path)
                        if method == 'sketch':
                            res = sketch_effect(img, 'gray', blur, 0, edge_low, edge_high)
                        elif method == 'oil':
                            res = oil_painting_effect(img, 'bilateral', strength, quant, True)
                        elif method == 'cartoon':
                            res = cartoon_effect(img, bi_iters, edge_low, edge_high)
                        elif method == 'color_lab' and style is not None:
                            res = color_transfer_lab(img, style)
                        elif method == 'hist_match' and style is not None:
                            res = histogram_match_rgb(img, style)
                        elif method == 'texture_blend' and style is not None:
                            res = pyramid_texture_blend(img, style, 4)
                        else:
                            res = img
                        save_image(os.path.join(out_dir, name), res)
                    except Exception as e:
                        print('处理失败:', path, e)
            messagebox.showinfo('完成', '批量处理完成')

        threading.Thread(target=worker, daemon=True).start()


if __name__ == '__main__':
    app = ImageApp()
    app.mainloop()

