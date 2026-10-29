"""Build an offline listening worksheet, with optional actual model predictions."""
import argparse
from html import escape
import json
import os
from pathlib import Path
from score_predictions import read_jsonl

CSS = '''body{font:16px/1.6 system-ui,sans-serif;background:#f3f6fa;color:#172238;margin:0}
main{max-width:1180px;margin:auto;padding:36px 24px}h1{font-size:30px;line-height:1.35}
article,.note{background:white;border:1px solid #dce3ed;border-radius:14px;padding:24px;margin:24px 0}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;background:#e6eefb;font-size:13px}
.warning{background:#fff3d6;padding:16px;border-radius:8px}.muted{color:#66738a;font-size:14px}
table{border-collapse:collapse;width:100%;margin-top:16px;font-size:14px}td,th{border-bottom:1px solid #e2e8f0;padding:10px;text-align:left}
th{background:#f5f7fb}tr.focus{background:#eef5ff}audio{width:100%;max-width:700px;margin:12px 0}
button{border:1px solid #b9c9e3;border-radius:6px;padding:6px 10px;background:#fff;cursor:pointer;white-space:nowrap}
button:hover{background:#e6eefb}textarea{width:95%;min-height:55px;display:block;margin:10px 0}
select{padding:6px}.scroll{overflow:auto}.ok{color:#00745c}.bad{color:#b32424}a{color:#2553a6}
@media(max-width:650px){main{padding:16px}article{padding:14px}h1{font-size:25px}}'''

JS = r'''let stops=new WeakMap();
function playPart(id,start,end){const a=document.getElementById(id); document.querySelectorAll('audio').forEach(x=>x.pause());
 if(stops.has(a))a.removeEventListener('timeupdate',stops.get(a));a.currentTime=start;
 const stop=()=>{if(a.currentTime>=end){a.pause();a.removeEventListener('timeupdate',stop);}};
 stops.set(a,stop);a.addEventListener('timeupdate',stop);a.play();}
function exportReview(){let rows=[['clip_id','review_status','review_notes']];
 document.querySelectorAll('article').forEach(x=>rows.push([x.dataset.clip,x.querySelector('select').value,x.querySelector('textarea').value]));
 const csv='\ufeff'+rows.map(r=>r.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\r\n');
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
 a.download='listening_review.csv';a.click();URL.revokeObjectURL(a.href);}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ami/prepared'))
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--output', type=Path, help='Optional separate worksheet; relative audio links are adjusted.')
    args = parser.parse_args()
    output = args.output or args.data / 'review.html'
    output.parent.mkdir(parents=True, exist_ok=True)

    def asset(path):
        return escape(os.path.relpath((args.data / path).resolve(), output.parent.resolve()).replace('\\', '/'))
    inputs = {r['clip_id']: r for r in read_jsonl(args.data / 'inputs.jsonl')}
    labels = read_jsonl(args.data / 'labels.jsonl')
    predictions = {r['clip_id']: {p['segment_id']: p for p in r['predictions']}
                   for r in read_jsonl(args.predictions)} if args.predictions else {}
    parts = ['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>AMI 대화 행동 전환 · 청취 검토</title><style>' + CSS + '</style><main>',
             '<span class="badge">자연 대화 · 시작용 표본 8개</span><h1>질문하던 사람이 답변해도<br>같은 사람인가?</h1>',
             '<p>원래 순서를 유지한 연속 대화입니다. 파란 행은 지정한 질문·응답 4개입니다. 같은 화자의 전후 발화를 비교하고, AMI 주석과 청취가 일치하는지 검토하세요.</p>',
             '<p class="warning"><strong>현재 모든 후보는 청취 검토 대기입니다.</strong> 자동 선정 이벤트와 모델 점수는 검토 완료를 의미하지 않습니다. '
             '화자 참조·정답 발화 경계를 제공한 과제이며 일반 DER 결과가 아닙니다. 이 페이지는 제출 홈페이지가 아닌 작업용 검토표입니다.</p>',
             '<button onclick="exportReview()">검토 결과 CSV 저장</button><p class="muted">선택·메모는 CSV로 저장해야 보존됩니다. 기존 review.csv와 정답 파일은 자동 변경하지 않습니다.</p>']
    for index, row in enumerate(labels):
        clip = row['clip_id']
        public = inputs[clip]
        by_id = {u['segment_id']: u for u in row['utterances']}
        times = {s['segment_id']: s for s in public['segments']}
        audio_id = f'audio_{index}'
        event = '질문↔답변 전환' if row['event'] == 'question_answer_reversal' else '선정 질문·응답 방향 유지'
        parts.extend([f'<article data-clip="{escape(clip)}"><span class="badge">{escape(row["split"])} · {event}</span>',
                      f'<h2>{escape(row["meeting"])} · {len(public["registered_speakers"])}명</h2>',
                      f'<div class="muted">원본 회의 {row["source_start"]:.2f}–{row["source_end"]:.2f}초 · {escape(clip)}</div>',
                      f'<audio id="{audio_id}" controls preload="none" src="{asset(public["audio"])}"></audio><ul>'])
        for pair in row['same_speaker_pairs']:
            first, later = by_id[pair['before_segment_id']], by_id[pair['after_segment_id']]
            eligible = '점수 포함' if pair['score_eligible'] else '점수 제외: 길이·겹침·잘림 조건'
            parts.append(f'<li>화자 <b>{pair["speaker"]}</b> · {escape(pair["transition"])} · '
                         f'{times[first["segment_id"]]["start"]:.2f}s → {times[later["segment_id"]]["start"]:.2f}s · {eligible}</li>')
        parts.append('</ul><details><summary>등록 음성 듣기</summary>')
        for speaker, refs in public['registered_speakers'].items():
            for ref in refs:
                parts.append(f'<div>화자 {speaker} · {escape(Path(ref["path"]).name)}<br><audio controls preload="none" src="{asset(ref["path"])}"></audio></div>')
        parts.append('</details><div class="scroll"><table><thead><tr><th>구간</th><th>화자 / 예측</th><th>대화 행동 · 전사</th><th>채점</th></tr></thead><tbody>')
        for u in row['utterances']:
            segment = times[u['segment_id']]
            pred = predictions.get(clip, {}).get(u['segment_id'], {})
            guessed = pred.get('speaker')
            predicted = f' / {guessed or "미예측"}' if predictions else ''
            css = 'ok' if guessed == u['speaker'] else 'bad'
            focus = ' class="focus"' if u.get('focus_slot') else ''
            duration = segment['end'] - segment['start']
            score = '포함' if u['score_eligible'] else '제외'
            detail = f'{duration:.2f}s · 겹침 {u["other_speaker_word_overlap_fraction"]:.0%}'
            parts.append(f'<tr{focus}><td><button onclick="playPart(\'{audio_id}\',{segment["start"]:.6f},{segment["end"]:.6f})">'
                         f'{segment["start"]:.2f}–{segment["end"]:.2f}s ▶</button></td>'
                         f'<td><b>{u["speaker"]}</b><span class="{css}">{escape(predicted)}</span></td>'
                         f'<td><span class="muted">{escape(u["dialogue_act"])} · {escape(u.get("focus_slot", "context"))}</span><br>{escape(u["text"])}</td>'
                         f'<td>{score}<br><span class="muted">{detail}</span></td></tr>')
        parts.append('</tbody></table></div><p>직접 청취 후 선택: <select><option value="pending">미검토</option>'
                     '<option value="accepted_after_listening">청취 후 채택</option><option value="needs_revision">시각·연결·화자 수정 필요</option>'
                     '<option value="rejected_after_listening">제외</option></select></p><textarea aria-label="검토 메모" placeholder="질문·응답 연결, 화자, 겹침, 간접 표현의 적절성 등"></textarea></article>')
    parts.append('<p class="muted">출처: AMI Corpus · CC BY 4.0. LICENCE_AMI.txt와 ATTRIBUTION.md 참조. 모델 예측은 해당 실행의 실제 출력이며 정답 검토를 대체하지 않습니다.</p></main><script>' + JS + '</script></html>')
    output.write_text('\n'.join(parts), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
