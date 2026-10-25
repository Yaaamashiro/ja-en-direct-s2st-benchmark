"""Lazy ipywidgets UI. Its view never receives a corpus/split/ID/score mapping."""
import html

from .text_pair_audit import LABELS, FLAGS, CONFIDENCES, report

LABEL_HELP = {
    'correct': '意味が実質的に一致。自然な言い換えは可。',
    'acceptable': '中心的な意味は一致。軽微な省略・追加・意訳は許容。',
    'partial_mismatch': '一部は対応するが、重要な欠落・追加・文境界のずれがある。',
    'wrong_alignment': '別内容、または意味が逆で、対応文として不適切。',
    'uncertain': '専門知識や文脈が足りず、自信をもって判定できない。',
}


class AuditUI:
    def __init__(self, store):
        import ipywidgets as w
        self.store, self.w = store, w
        self.position, self.editing, self.busy = store.first_pending(), False, False
        self.closed = False
        self.heading = w.HTML()
        self.japanese = w.Textarea(description='Japanese:', disabled=True, layout=w.Layout(width='100%', height='110px'))
        self.english = w.Textarea(description='English:', disabled=True, layout=w.Layout(width='100%', height='110px'))
        self.raw = w.HTML()
        self.provenance = w.Accordion(children=[self.raw], selected_index=None)
        self.provenance.set_title(0, 'Show provenance (text only)')
        self.buttons = {name: w.Button(description=name.replace('_', ' ').title(), tooltip=LABEL_HELP[name],
            layout=w.Layout(width='auto')) for name in LABELS}
        self.flags = {name: w.Checkbox(description=name, indent=False, layout=w.Layout(width='300px')) for name in FLAGS}
        self.confidence = w.Dropdown(description='Confidence:', options=CONFIDENCES, value='high')
        self.label = w.Dropdown(description='Label:', options=[('Choose a human label', None), *[(k, k) for k in LABELS]])
        self.note = w.Textarea(description='Note:', layout=w.Layout(width='100%'))
        self.previous = w.Button(description='Previous')
        self.next = w.Button(description='Save & Next', button_style='primary')
        self.message = w.HTML()
        self.output = w.Output()
        guide = w.HTML('<ul>'+''.join('<li><b>'+name+'</b>: '+html.escape(LABEL_HELP[name])+'</li>' for name in LABELS)+'</ul>')
        self.widget = w.VBox([self.heading, self.japanese, self.english, self.provenance, guide,
            w.HBox(list(self.buttons.values())), self.label,
            w.VBox(list(self.flags.values())), self.confidence, self.note,
            w.HBox([self.previous, self.next]), self.message, self.output])
        for name, button in self.buttons.items():
            button.on_click(lambda _, label=name: self.save_next(label))
        self.next.on_click(lambda _: self.save_next(self.label.value))
        self.previous.on_click(lambda _: self.go_previous())
        self.render()

    def close(self):
        self.closed = True
        for button in (*self.buttons.values(), self.previous, self.next):
            button.disabled = True

    def render(self):
        done = len(self.store.annotations)
        total = len(self.store.queue)
        self.message.value = ''
        self.provenance.selected_index = None
        self.output.clear_output(wait=True)
        at_end = self.position >= total
        self.heading.value = ('<b>Annotation complete</b>' if at_end else f'<b>Sample {self.position+1} / {total}</b>')
        self.heading.value += f'<br>Annotated: {done} / {total} — Remaining: {total-done}'
        self.previous.disabled = self.closed or self.position == 0
        for widget in (*self.buttons.values(), self.next, self.label, self.confidence, self.note, *self.flags.values()):
            widget.disabled = self.closed or at_end
        if at_end:
            self.japanese.value = self.english.value = self.raw.value = ''
            if done == total:
                from IPython.display import display, Markdown
                with self.output:
                    display(Markdown(report(self.store.publish())))
            return
        sample = self.store.blind_sample(self.position)
        self.japanese.value, self.english.value = sample['ja_text'], sample['en_text']
        self.raw.value = ''.join('<b>'+html.escape(key)+'</b><pre>'+html.escape(value)+'</pre>'
                                 for key, value in sample['provenance'].items())
        row = self.store.annotations.get(self.store.queue[self.position]['audit_id'])
        self.label.value = row['label'] if row else None
        self.confidence.value = row['confidence'] if row else 'high'
        self.note.value = row['note'] if row else ''
        for name, widget in self.flags.items():
            widget.value = bool(row and name in row['flags'])
        if row:
            self.message.value = '保存済みの判定を表示しています。変更して保存した場合だけ更新します。'

    def save_next(self, label):
        if self.closed or self.busy or self.position >= len(self.store.queue):
            return
        self.busy = True
        row = self.store.queue[self.position]
        before = self.store.annotations.get(row['audit_id'])
        for button in (*self.buttons.values(), self.previous, self.next):
            button.disabled = True
        try:
            self.store.save(row['audit_id'], label, flags=[k for k, v in self.flags.items() if v.value],
                confidence=self.confidence.value, note=self.note.value, allow_update=self.editing)
            self.position += 1
            while self.position < len(self.store.queue) and self.store.queue[self.position]['audit_id'] in self.store.annotations:
                self.position += 1
            self.editing = False
            self.render()
        except Exception as error:
            # Keep the same sample and controls so failed saves can be retried.
            if self.store.annotations.get(row['audit_id']) != before:
                self.message.value = '<b>判定は保存済みですが、集計表示の更新に失敗しました。セル8Dを再実行して復旧してください。</b><br>'+html.escape(str(error))
                self.close()
            else:
                self.message.value = '<b>保存できませんでした。次へ進んでいません。</b><br>'+html.escape(str(error))
                for button in (*self.buttons.values(), self.previous, self.next):
                    button.disabled = False
        finally:
            self.busy = False

    def go_previous(self):
        if self.closed or self.busy or self.position <= 0:
            return
        self.position -= 1
        self.editing = True
        self.render()


def show(store):
    from IPython.display import display
    ui = AuditUI(store)
    display(ui.widget)
    return ui
