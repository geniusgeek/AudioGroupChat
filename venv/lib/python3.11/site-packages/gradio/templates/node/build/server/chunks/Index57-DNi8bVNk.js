import { c as create_ssr_component, v as validate_component, a as createEventDispatcher, e as escape, b as add_attribute, f as each, n as null_to_empty, o as onDestroy } from './ssr-1WRCSaMf.js';
import { t as tick } from './Component-CIvFpclA.js';
import { B as Block, S as Static, x as MarkdownCode, z as Copy, V as Minimize, W as Maximize } from './2-BBGSu8z1.js';
import { d as dequal } from './index6-sfNUnwRZ.js';
import { U as Upload } from './ModifyUpload-C5mToh5s.js';
export { default as BaseExample } from './Example8-Omf41Jmj.js';
import './index-BzvnFJlw.js';
import 'tty';
import 'path';
import 'url';
import 'fs';

const css$5 = {
  code: "input.svelte-1xjy58k{position:absolute;top:var(--size-2);right:var(--size-2);bottom:var(--size-2);left:var(--size-2);flex:1 1 0%;transform:translateX(-0.1px);outline:none;border:none;background:transparent;cursor:text}span.svelte-1xjy58k{flex:1 1 0%;position:relative;display:inline-block;outline:none;-webkit-user-select:text;-moz-user-select:text;-ms-user-select:text;user-select:text;cursor:text;width:100%;height:100%}span.expanded.svelte-1xjy58k{height:auto;min-height:100%;white-space:pre-wrap;word-break:break-word;white-space:normal}.multiline.svelte-1xjy58k{white-space:pre-line}.header.svelte-1xjy58k{transform:translateX(0);font-weight:var(--weight-bold);white-space:normal;word-break:break-word;margin-left:var(--size-1)}.edit.svelte-1xjy58k{opacity:0;pointer-events:none}",
  map: '{"version":3,"file":"EditableCell.svelte","sources":["EditableCell.svelte"],"sourcesContent":["<script lang=\\"ts\\">import { createEventDispatcher } from \\"svelte\\";\\nimport { MarkdownCode } from \\"@gradio/markdown-code\\";\\nexport let edit;\\nexport let value = \\"\\";\\nexport let display_value = null;\\nexport let styling = \\"\\";\\nexport let header = false;\\nexport let datatype = \\"str\\";\\nexport let latex_delimiters;\\nexport let clear_on_focus = false;\\nexport let line_breaks = true;\\nexport let editable = true;\\nexport let root;\\nexport let max_chars = null;\\nconst dispatch = createEventDispatcher();\\nlet is_expanded = false;\\nexport let el;\\n$: _value = value;\\nfunction truncate_text(text, max_length = null) {\\n    const str = String(text);\\n    if (!max_length || str.length <= max_length)\\n        return str;\\n    return str.slice(0, max_length) + \\"...\\";\\n}\\n$: display_text = is_expanded ? value : truncate_text(display_value || value, max_chars);\\nfunction use_focus(node) {\\n    if (clear_on_focus) {\\n        _value = \\"\\";\\n    }\\n    requestAnimationFrame(() => {\\n        node.focus();\\n    });\\n    return {};\\n}\\nfunction handle_blur({ currentTarget }) {\\n    value = currentTarget.value;\\n    dispatch(\\"blur\\");\\n}\\nfunction handle_keydown(event) {\\n    if (event.key === \\"Enter\\") {\\n        if (edit) {\\n            value = _value;\\n            dispatch(\\"blur\\");\\n        }\\n        else if (!header) {\\n            is_expanded = !is_expanded;\\n        }\\n    }\\n    dispatch(\\"keydown\\", event);\\n}\\nfunction handle_click() {\\n    if (!edit && !header) {\\n        is_expanded = !is_expanded;\\n    }\\n}\\n<\/script>\\n\\n{#if edit}\\n\\t<input\\n\\t\\trole=\\"textbox\\"\\n\\t\\tbind:this={el}\\n\\t\\tbind:value={_value}\\n\\t\\tclass:header\\n\\t\\ttabindex=\\"-1\\"\\n\\t\\ton:blur={handle_blur}\\n\\t\\ton:mousedown|stopPropagation\\n\\t\\ton:mouseup|stopPropagation\\n\\t\\ton:click|stopPropagation\\n\\t\\tuse:use_focus\\n\\t\\ton:keydown={handle_keydown}\\n\\t/>\\n{/if}\\n\\n<span\\n\\ton:click={handle_click}\\n\\ton:keydown={handle_keydown}\\n\\ttabindex=\\"0\\"\\n\\trole=\\"button\\"\\n\\tclass:edit\\n\\tclass:expanded={is_expanded}\\n\\tclass:multiline={header}\\n\\ton:focus|preventDefault\\n\\tstyle={styling}\\n\\tdata-editable={editable}\\n\\tplaceholder=\\" \\"\\n>\\n\\t{#if datatype === \\"html\\"}\\n\\t\\t{@html display_text}\\n\\t{:else if datatype === \\"markdown\\"}\\n\\t\\t<MarkdownCode\\n\\t\\t\\tmessage={display_text.toLocaleString()}\\n\\t\\t\\t{latex_delimiters}\\n\\t\\t\\t{line_breaks}\\n\\t\\t\\tchatbot={false}\\n\\t\\t\\t{root}\\n\\t\\t/>\\n\\t{:else}\\n\\t\\t{editable ? display_text : display_value || display_text}\\n\\t{/if}\\n</span>\\n\\n<style>\\n\\tinput {\\n\\t\\tposition: absolute;\\n\\t\\ttop: var(--size-2);\\n\\t\\tright: var(--size-2);\\n\\t\\tbottom: var(--size-2);\\n\\t\\tleft: var(--size-2);\\n\\t\\tflex: 1 1 0%;\\n\\t\\ttransform: translateX(-0.1px);\\n\\t\\toutline: none;\\n\\t\\tborder: none;\\n\\t\\tbackground: transparent;\\n\\t\\tcursor: text;\\n\\t}\\n\\n\\tspan {\\n\\t\\tflex: 1 1 0%;\\n\\t\\tposition: relative;\\n\\t\\tdisplay: inline-block;\\n\\t\\toutline: none;\\n\\t\\t-webkit-user-select: text;\\n\\t\\t-moz-user-select: text;\\n\\t\\t-ms-user-select: text;\\n\\t\\tuser-select: text;\\n\\t\\tcursor: text;\\n\\t\\twidth: 100%;\\n\\t\\theight: 100%;\\n\\t}\\n\\n\\tspan.expanded {\\n\\t\\theight: auto;\\n\\t\\tmin-height: 100%;\\n\\t\\twhite-space: pre-wrap;\\n\\t\\tword-break: break-word;\\n\\t\\twhite-space: normal;\\n\\t}\\n\\n\\t.multiline {\\n\\t\\twhite-space: pre-line;\\n\\t}\\n\\n\\t.header {\\n\\t\\ttransform: translateX(0);\\n\\t\\tfont-weight: var(--weight-bold);\\n\\t\\twhite-space: normal;\\n\\t\\tword-break: break-word;\\n\\t\\tmargin-left: var(--size-1);\\n\\t}\\n\\n\\t.edit {\\n\\t\\topacity: 0;\\n\\t\\tpointer-events: none;\\n\\t}</style>\\n"],"names":[],"mappings":"AAsGC,oBAAM,CACL,QAAQ,CAAE,QAAQ,CAClB,GAAG,CAAE,IAAI,QAAQ,CAAC,CAClB,KAAK,CAAE,IAAI,QAAQ,CAAC,CACpB,MAAM,CAAE,IAAI,QAAQ,CAAC,CACrB,IAAI,CAAE,IAAI,QAAQ,CAAC,CACnB,IAAI,CAAE,CAAC,CAAC,CAAC,CAAC,EAAE,CACZ,SAAS,CAAE,WAAW,MAAM,CAAC,CAC7B,OAAO,CAAE,IAAI,CACb,MAAM,CAAE,IAAI,CACZ,UAAU,CAAE,WAAW,CACvB,MAAM,CAAE,IACT,CAEA,mBAAK,CACJ,IAAI,CAAE,CAAC,CAAC,CAAC,CAAC,EAAE,CACZ,QAAQ,CAAE,QAAQ,CAClB,OAAO,CAAE,YAAY,CACrB,OAAO,CAAE,IAAI,CACb,mBAAmB,CAAE,IAAI,CACzB,gBAAgB,CAAE,IAAI,CACtB,eAAe,CAAE,IAAI,CACrB,WAAW,CAAE,IAAI,CACjB,MAAM,CAAE,IAAI,CACZ,KAAK,CAAE,IAAI,CACX,MAAM,CAAE,IACT,CAEA,IAAI,wBAAU,CACb,MAAM,CAAE,IAAI,CACZ,UAAU,CAAE,IAAI,CAChB,WAAW,CAAE,QAAQ,CACrB,UAAU,CAAE,UAAU,CACtB,WAAW,CAAE,MACd,CAEA,yBAAW,CACV,WAAW,CAAE,QACd,CAEA,sBAAQ,CACP,SAAS,CAAE,WAAW,CAAC,CAAC,CACxB,WAAW,CAAE,IAAI,aAAa,CAAC,CAC/B,WAAW,CAAE,MAAM,CACnB,UAAU,CAAE,UAAU,CACtB,WAAW,CAAE,IAAI,QAAQ,CAC1B,CAEA,oBAAM,CACL,OAAO,CAAE,CAAC,CACV,cAAc,CAAE,IACjB"}'
};
function truncate_text(text, max_length = null) {
  const str = String(text);
  if (!max_length || str.length <= max_length)
    return str;
  return str.slice(0, max_length) + "...";
}
const EditableCell = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let _value;
  let display_text;
  let { edit } = $$props;
  let { value = "" } = $$props;
  let { display_value = null } = $$props;
  let { styling = "" } = $$props;
  let { header = false } = $$props;
  let { datatype = "str" } = $$props;
  let { latex_delimiters } = $$props;
  let { clear_on_focus = false } = $$props;
  let { line_breaks = true } = $$props;
  let { editable = true } = $$props;
  let { root } = $$props;
  let { max_chars = null } = $$props;
  createEventDispatcher();
  let { el } = $$props;
  if ($$props.edit === void 0 && $$bindings.edit && edit !== void 0)
    $$bindings.edit(edit);
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  if ($$props.display_value === void 0 && $$bindings.display_value && display_value !== void 0)
    $$bindings.display_value(display_value);
  if ($$props.styling === void 0 && $$bindings.styling && styling !== void 0)
    $$bindings.styling(styling);
  if ($$props.header === void 0 && $$bindings.header && header !== void 0)
    $$bindings.header(header);
  if ($$props.datatype === void 0 && $$bindings.datatype && datatype !== void 0)
    $$bindings.datatype(datatype);
  if ($$props.latex_delimiters === void 0 && $$bindings.latex_delimiters && latex_delimiters !== void 0)
    $$bindings.latex_delimiters(latex_delimiters);
  if ($$props.clear_on_focus === void 0 && $$bindings.clear_on_focus && clear_on_focus !== void 0)
    $$bindings.clear_on_focus(clear_on_focus);
  if ($$props.line_breaks === void 0 && $$bindings.line_breaks && line_breaks !== void 0)
    $$bindings.line_breaks(line_breaks);
  if ($$props.editable === void 0 && $$bindings.editable && editable !== void 0)
    $$bindings.editable(editable);
  if ($$props.root === void 0 && $$bindings.root && root !== void 0)
    $$bindings.root(root);
  if ($$props.max_chars === void 0 && $$bindings.max_chars && max_chars !== void 0)
    $$bindings.max_chars(max_chars);
  if ($$props.el === void 0 && $$bindings.el && el !== void 0)
    $$bindings.el(el);
  $$result.css.add(css$5);
  _value = value;
  display_text = truncate_text(display_value || value, max_chars);
  return `${edit ? `<input role="textbox" tabindex="-1" class="${["svelte-1xjy58k", header ? "header" : ""].join(" ").trim()}"${add_attribute("this", el, 0)}${add_attribute("value", _value, 0)}>` : ``} <span tabindex="0" role="button"${add_attribute("style", styling, 0)}${add_attribute("data-editable", editable, 0)} placeholder=" " class="${[
    "svelte-1xjy58k",
    (edit ? "edit" : "") + "  " + (header ? "multiline" : "")
  ].join(" ").trim()}">${datatype === "html" ? `<!-- HTML_TAG_START -->${display_text}<!-- HTML_TAG_END -->` : `${datatype === "markdown" ? `${validate_component(MarkdownCode, "MarkdownCode").$$render(
    $$result,
    {
      message: display_text.toLocaleString(),
      latex_delimiters,
      line_breaks,
      chatbot: false,
      root
    },
    {},
    {}
  )}` : `${escape(editable ? display_text : display_value || display_text)}`}`} </span>`;
});
const css$4 = {
  code: "table.svelte-y11bhb.svelte-y11bhb{position:relative;overflow-y:scroll;overflow-x:scroll;-webkit-overflow-scrolling:touch;max-height:var(--max-height);box-sizing:border-box;display:block;padding:0;margin:0;color:var(--body-text-color);font-size:var(--input-text-size);line-height:var(--line-md);font-family:var(--font-mono);border-spacing:0;width:100%;scroll-snap-type:x proximity;border-collapse:separate}table.svelte-y11bhb .svelte-y11bhb:is(thead, tfoot, tbody){display:table;table-layout:fixed;width:100%;box-sizing:border-box}tbody.svelte-y11bhb.svelte-y11bhb{overflow-x:scroll;overflow-y:hidden}table.svelte-y11bhb tbody.svelte-y11bhb{padding-top:var(--bw-svt-p-top);padding-bottom:var(--bw-svt-p-bottom)}tbody.svelte-y11bhb.svelte-y11bhb{position:relative;box-sizing:border-box;border:0px solid currentColor}tbody.svelte-y11bhb>tr:last-child{border:none}table.svelte-y11bhb td{scroll-snap-align:start}tbody.svelte-y11bhb>tr:nth-child(even){background:var(--table-even-background-fill)}tbody.svelte-y11bhb td.frozen-column{position:sticky;z-index:var(--layer-2)}tbody.svelte-y11bhb tr:nth-child(odd) td.frozen-column{background:var(--table-odd-background-fill)}tbody.svelte-y11bhb tr:nth-child(even) td.frozen-column{background:var(--table-even-background-fill)}tbody.svelte-y11bhb td.always-frozen{z-index:var(--layer-3)}tbody.svelte-y11bhb td.last-frozen{border-right:2px solid var(--border-color-primary)}thead.svelte-y11bhb.svelte-y11bhb{position:sticky;top:0;left:0;z-index:var(--layer-3);background:var(--background-fill-primary)}thead.svelte-y11bhb th{background:var(--table-even-background-fill) !important}thead.svelte-y11bhb th.frozen-column{position:sticky;z-index:var(--layer-4)}thead.svelte-y11bhb th.always-frozen{z-index:var(--layer-5)}.table.disable-scroll.svelte-y11bhb.svelte-y11bhb{overflow:hidden !important}",
  map: '{"version":3,"file":"VirtualTable.svelte","sources":["VirtualTable.svelte"],"sourcesContent":["<script lang=\\"ts\\">import { onMount, tick } from \\"svelte\\";\\nimport { _ } from \\"svelte-i18n\\";\\nexport let items = [];\\nexport let max_height;\\nexport let actual_height;\\nexport let table_scrollbar_width;\\nexport let start = 0;\\nexport let end = 20;\\nexport let selected;\\nexport let disable_scroll = false;\\nlet height = \\"100%\\";\\nlet average_height = 30;\\nlet bottom = 0;\\nlet contents;\\nlet head_height = 0;\\nlet foot_height = 0;\\nlet height_map = [];\\nlet mounted;\\nlet rows;\\nlet top = 0;\\nlet viewport;\\nlet viewport_height = 200;\\nlet visible = [];\\nlet viewport_box;\\n$: viewport_height = viewport_box?.height || 200;\\nconst is_browser = typeof window !== \\"undefined\\";\\nconst raf = is_browser ? window.requestAnimationFrame : (cb) => cb();\\n$: mounted && raf(() => refresh_height_map(sortedItems));\\nlet content_height = 0;\\nasync function refresh_height_map(_items) {\\n    if (viewport_height === 0) {\\n        return;\\n    }\\n    head_height = viewport.querySelector(\\".thead\\")?.getBoundingClientRect().height || 0;\\n    await tick();\\n    const { scrollTop } = viewport;\\n    table_scrollbar_width = viewport.offsetWidth - viewport.clientWidth;\\n    content_height = top - (scrollTop - head_height);\\n    let i = start;\\n    while (content_height < max_height && i < _items.length) {\\n        let row = rows[i - start];\\n        if (!row) {\\n            end = i + 1;\\n            await tick();\\n            row = rows[i - start];\\n        }\\n        let _h = row?.getBoundingClientRect().height;\\n        if (!_h) {\\n            _h = average_height;\\n        }\\n        const row_height = height_map[i] = _h;\\n        content_height += row_height;\\n        i += 1;\\n    }\\n    end = i;\\n    const remaining = _items.length - end;\\n    const scrollbar_height = viewport.offsetHeight - viewport.clientHeight;\\n    if (scrollbar_height > 0) {\\n        content_height += scrollbar_height;\\n    }\\n    let filtered_height_map = height_map.filter((v) => typeof v === \\"number\\");\\n    average_height = filtered_height_map.reduce((a, b) => a + b, 0) / filtered_height_map.length;\\n    bottom = remaining * average_height;\\n    height_map.length = _items.length;\\n    await tick();\\n    if (!max_height) {\\n        actual_height = content_height + 1;\\n    }\\n    else if (content_height < max_height) {\\n        actual_height = content_height + 2;\\n    }\\n    else {\\n        actual_height = max_height;\\n    }\\n    await tick();\\n}\\n$: scroll_and_render(selected);\\nasync function scroll_and_render(n) {\\n    raf(async () => {\\n        if (typeof n !== \\"number\\")\\n            return;\\n        const direction = typeof n !== \\"number\\" ? false : is_in_view(n);\\n        if (direction === true) {\\n            return;\\n        }\\n        if (direction === \\"back\\") {\\n            await scroll_to_index(n, { behavior: \\"instant\\" });\\n        }\\n        if (direction === \\"forwards\\") {\\n            await scroll_to_index(n, { behavior: \\"instant\\" }, true);\\n        }\\n    });\\n}\\nfunction is_in_view(n) {\\n    const current = rows && rows[n - start];\\n    if (!current && n < start) {\\n        return \\"back\\";\\n    }\\n    if (!current && n >= end - 1) {\\n        return \\"forwards\\";\\n    }\\n    const { top: viewport_top } = viewport.getBoundingClientRect();\\n    const { top: top2, bottom: bottom2 } = current.getBoundingClientRect();\\n    if (top2 - viewport_top < 37) {\\n        return \\"back\\";\\n    }\\n    if (bottom2 - viewport_top > viewport_height) {\\n        return \\"forwards\\";\\n    }\\n    return true;\\n}\\nfunction get_computed_px_amount(elem, property) {\\n    if (!elem) {\\n        return 0;\\n    }\\n    const compStyle = getComputedStyle(elem);\\n    let x = parseInt(compStyle.getPropertyValue(property));\\n    return x;\\n}\\nasync function handle_scroll(e) {\\n    const scroll_top = viewport.scrollTop;\\n    rows = contents.children;\\n    const is_start_overflow = sortedItems.length < start;\\n    const row_top_border = get_computed_px_amount(rows[1], \\"border-top-width\\");\\n    const actual_border_collapsed_width = 0;\\n    if (is_start_overflow) {\\n        await scroll_to_index(sortedItems.length - 1, { behavior: \\"auto\\" });\\n    }\\n    let new_start = 0;\\n    for (let v = 0; v < rows.length; v += 1) {\\n        height_map[start + v] = rows[v].getBoundingClientRect().height;\\n    }\\n    let i = 0;\\n    let y = head_height + row_top_border / 2;\\n    let row_heights = [];\\n    while (i < sortedItems.length) {\\n        const row_height = height_map[i] || average_height;\\n        row_heights[i] = row_height;\\n        if (y + row_height + actual_border_collapsed_width > scroll_top) {\\n            new_start = i;\\n            top = y - (head_height + row_top_border / 2);\\n            break;\\n        }\\n        y += row_height;\\n        i += 1;\\n    }\\n    new_start = Math.max(0, new_start);\\n    while (i < sortedItems.length) {\\n        const row_height = height_map[i] || average_height;\\n        y += row_height;\\n        i += 1;\\n        if (y > scroll_top + viewport_height) {\\n            break;\\n        }\\n    }\\n    start = new_start;\\n    end = i;\\n    const remaining = sortedItems.length - end;\\n    if (end === 0) {\\n        end = 10;\\n    }\\n    average_height = (y - head_height) / end;\\n    let remaining_height = remaining * average_height;\\n    while (i < sortedItems.length) {\\n        i += 1;\\n        height_map[i] = average_height;\\n    }\\n    bottom = remaining_height;\\n    if (!isFinite(bottom)) {\\n        bottom = 2e5;\\n    }\\n}\\nexport async function scroll_to_index(index, opts, align_end = false) {\\n    await tick();\\n    const _itemHeight = average_height;\\n    let distance = index * _itemHeight;\\n    if (align_end) {\\n        distance = distance - viewport_height + _itemHeight + head_height;\\n    }\\n    const scrollbar_height = viewport.offsetHeight - viewport.clientHeight;\\n    if (scrollbar_height > 0) {\\n        distance += scrollbar_height;\\n    }\\n    const _opts = {\\n        top: distance,\\n        behavior: \\"smooth\\",\\n        ...opts\\n    };\\n    viewport.scrollTo(_opts);\\n}\\n$: sortedItems = items;\\n$: visible = is_browser ? sortedItems.slice(start, end).map((data, i) => {\\n    return { index: i + start, data };\\n}) : sortedItems.slice(0, max_height / sortedItems.length * average_height + 1).map((data, i) => {\\n    return { index: i + start, data };\\n});\\nonMount(() => {\\n    rows = contents.children;\\n    mounted = true;\\n    refresh_height_map(items);\\n});\\n<\/script>\\n\\n<svelte-virtual-table-viewport>\\n\\t<div>\\n\\t\\t<table\\n\\t\\t\\tclass=\\"table\\"\\n\\t\\t\\tclass:disable-scroll={disable_scroll}\\n\\t\\t\\tbind:this={viewport}\\n\\t\\t\\tbind:contentRect={viewport_box}\\n\\t\\t\\ton:scroll={handle_scroll}\\n\\t\\t\\tstyle=\\"height: {height}; --bw-svt-p-top: {top}px; --bw-svt-p-bottom: {bottom}px; --bw-svt-head-height: {head_height}px; --bw-svt-foot-height: {foot_height}px; --bw-svt-avg-row-height: {average_height}px; --max-height: {max_height}px\\"\\n\\t\\t>\\n\\t\\t\\t<thead class=\\"thead\\" bind:offsetHeight={head_height}>\\n\\t\\t\\t\\t<slot name=\\"thead\\" />\\n\\t\\t\\t</thead>\\n\\t\\t\\t<tbody bind:this={contents} class=\\"tbody\\">\\n\\t\\t\\t\\t{#if visible.length && visible[0].data.length}\\n\\t\\t\\t\\t\\t{#each visible as item (item.data[0].id)}\\n\\t\\t\\t\\t\\t\\t<slot name=\\"tbody\\" item={item.data} index={item.index}>\\n\\t\\t\\t\\t\\t\\t\\tMissing Table Row\\n\\t\\t\\t\\t\\t\\t</slot>\\n\\t\\t\\t\\t\\t{/each}\\n\\t\\t\\t\\t{/if}\\n\\t\\t\\t</tbody>\\n\\t\\t\\t<tfoot class=\\"tfoot\\" bind:offsetHeight={foot_height}>\\n\\t\\t\\t\\t<slot name=\\"tfoot\\" />\\n\\t\\t\\t</tfoot>\\n\\t\\t</table>\\n\\t</div>\\n</svelte-virtual-table-viewport>\\n\\n<style type=\\"text/css\\">\\n\\ttable {\\n\\t\\tposition: relative;\\n\\t\\toverflow-y: scroll;\\n\\t\\toverflow-x: scroll;\\n\\t\\t-webkit-overflow-scrolling: touch;\\n\\t\\tmax-height: var(--max-height);\\n\\t\\tbox-sizing: border-box;\\n\\t\\tdisplay: block;\\n\\t\\tpadding: 0;\\n\\t\\tmargin: 0;\\n\\t\\tcolor: var(--body-text-color);\\n\\t\\tfont-size: var(--input-text-size);\\n\\t\\tline-height: var(--line-md);\\n\\t\\tfont-family: var(--font-mono);\\n\\t\\tborder-spacing: 0;\\n\\t\\twidth: 100%;\\n\\t\\tscroll-snap-type: x proximity;\\n\\t\\tborder-collapse: separate;\\n\\t}\\n\\ttable :is(thead, tfoot, tbody) {\\n\\t\\tdisplay: table;\\n\\t\\ttable-layout: fixed;\\n\\t\\twidth: 100%;\\n\\t\\tbox-sizing: border-box;\\n\\t}\\n\\n\\ttbody {\\n\\t\\toverflow-x: scroll;\\n\\t\\toverflow-y: hidden;\\n\\t}\\n\\n\\ttable tbody {\\n\\t\\tpadding-top: var(--bw-svt-p-top);\\n\\t\\tpadding-bottom: var(--bw-svt-p-bottom);\\n\\t}\\n\\ttbody {\\n\\t\\tposition: relative;\\n\\t\\tbox-sizing: border-box;\\n\\t\\tborder: 0px solid currentColor;\\n\\t}\\n\\n\\ttbody > :global(tr:last-child) {\\n\\t\\tborder: none;\\n\\t}\\n\\n\\ttable :global(td) {\\n\\t\\tscroll-snap-align: start;\\n\\t}\\n\\n\\ttbody > :global(tr:nth-child(even)) {\\n\\t\\tbackground: var(--table-even-background-fill);\\n\\t}\\n\\n\\ttbody :global(td.frozen-column) {\\n\\t\\tposition: sticky;\\n\\t\\tz-index: var(--layer-2);\\n\\t}\\n\\n\\ttbody :global(tr:nth-child(odd)) :global(td.frozen-column) {\\n\\t\\tbackground: var(--table-odd-background-fill);\\n\\t}\\n\\n\\ttbody :global(tr:nth-child(even)) :global(td.frozen-column) {\\n\\t\\tbackground: var(--table-even-background-fill);\\n\\t}\\n\\n\\ttbody :global(td.always-frozen) {\\n\\t\\tz-index: var(--layer-3);\\n\\t}\\n\\n\\ttbody :global(td.last-frozen) {\\n\\t\\tborder-right: 2px solid var(--border-color-primary);\\n\\t}\\n\\n\\tthead {\\n\\t\\tposition: sticky;\\n\\t\\ttop: 0;\\n\\t\\tleft: 0;\\n\\t\\tz-index: var(--layer-3);\\n\\t\\tbackground: var(--background-fill-primary);\\n\\t}\\n\\n\\tthead :global(th) {\\n\\t\\tbackground: var(--table-even-background-fill) !important;\\n\\t}\\n\\n\\tthead :global(th.frozen-column) {\\n\\t\\tposition: sticky;\\n\\t\\tz-index: var(--layer-4);\\n\\t}\\n\\n\\tthead :global(th.always-frozen) {\\n\\t\\tz-index: var(--layer-5);\\n\\t}\\n\\n\\t.table.disable-scroll {\\n\\t\\toverflow: hidden !important;\\n\\t}</style>\\n"],"names":[],"mappings":"AAyOC,iCAAM,CACL,QAAQ,CAAE,QAAQ,CAClB,UAAU,CAAE,MAAM,CAClB,UAAU,CAAE,MAAM,CAClB,0BAA0B,CAAE,KAAK,CACjC,UAAU,CAAE,IAAI,YAAY,CAAC,CAC7B,UAAU,CAAE,UAAU,CACtB,OAAO,CAAE,KAAK,CACd,OAAO,CAAE,CAAC,CACV,MAAM,CAAE,CAAC,CACT,KAAK,CAAE,IAAI,iBAAiB,CAAC,CAC7B,SAAS,CAAE,IAAI,iBAAiB,CAAC,CACjC,WAAW,CAAE,IAAI,SAAS,CAAC,CAC3B,WAAW,CAAE,IAAI,WAAW,CAAC,CAC7B,cAAc,CAAE,CAAC,CACjB,KAAK,CAAE,IAAI,CACX,gBAAgB,CAAE,CAAC,CAAC,SAAS,CAC7B,eAAe,CAAE,QAClB,CACA,mBAAK,eAAC,IAAI,KAAK,EAAE,KAAK,EAAE,KAAK,CAAE,CAC9B,OAAO,CAAE,KAAK,CACd,YAAY,CAAE,KAAK,CACnB,KAAK,CAAE,IAAI,CACX,UAAU,CAAE,UACb,CAEA,iCAAM,CACL,UAAU,CAAE,MAAM,CAClB,UAAU,CAAE,MACb,CAEA,mBAAK,CAAC,mBAAM,CACX,WAAW,CAAE,IAAI,cAAc,CAAC,CAChC,cAAc,CAAE,IAAI,iBAAiB,CACtC,CACA,iCAAM,CACL,QAAQ,CAAE,QAAQ,CAClB,UAAU,CAAE,UAAU,CACtB,MAAM,CAAE,GAAG,CAAC,KAAK,CAAC,YACnB,CAEA,mBAAK,CAAW,aAAe,CAC9B,MAAM,CAAE,IACT,CAEA,mBAAK,CAAS,EAAI,CACjB,iBAAiB,CAAE,KACpB,CAEA,mBAAK,CAAW,kBAAoB,CACnC,UAAU,CAAE,IAAI,4BAA4B,CAC7C,CAEA,mBAAK,CAAS,gBAAkB,CAC/B,QAAQ,CAAE,MAAM,CAChB,OAAO,CAAE,IAAI,SAAS,CACvB,CAEA,mBAAK,CAAS,iBAAkB,CAAS,gBAAkB,CAC1D,UAAU,CAAE,IAAI,2BAA2B,CAC5C,CAEA,mBAAK,CAAS,kBAAmB,CAAS,gBAAkB,CAC3D,UAAU,CAAE,IAAI,4BAA4B,CAC7C,CAEA,mBAAK,CAAS,gBAAkB,CAC/B,OAAO,CAAE,IAAI,SAAS,CACvB,CAEA,mBAAK,CAAS,cAAgB,CAC7B,YAAY,CAAE,GAAG,CAAC,KAAK,CAAC,IAAI,sBAAsB,CACnD,CAEA,iCAAM,CACL,QAAQ,CAAE,MAAM,CAChB,GAAG,CAAE,CAAC,CACN,IAAI,CAAE,CAAC,CACP,OAAO,CAAE,IAAI,SAAS,CAAC,CACvB,UAAU,CAAE,IAAI,yBAAyB,CAC1C,CAEA,mBAAK,CAAS,EAAI,CACjB,UAAU,CAAE,IAAI,4BAA4B,CAAC,CAAC,UAC/C,CAEA,mBAAK,CAAS,gBAAkB,CAC/B,QAAQ,CAAE,MAAM,CAChB,OAAO,CAAE,IAAI,SAAS,CACvB,CAEA,mBAAK,CAAS,gBAAkB,CAC/B,OAAO,CAAE,IAAI,SAAS,CACvB,CAEA,MAAM,2CAAgB,CACrB,QAAQ,CAAE,MAAM,CAAC,UAClB"}'
};
let height = "100%";
const VirtualTable = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let sortedItems;
  let { items = [] } = $$props;
  let { max_height } = $$props;
  let { actual_height } = $$props;
  let { table_scrollbar_width } = $$props;
  let { start = 0 } = $$props;
  let { end = 20 } = $$props;
  let { selected } = $$props;
  let { disable_scroll = false } = $$props;
  let average_height = 30;
  let bottom = 0;
  let contents;
  let head_height = 0;
  let foot_height = 0;
  let rows;
  let top = 0;
  let viewport;
  let viewport_height = 200;
  let visible = [];
  const is_browser = typeof window !== "undefined";
  const raf = is_browser ? window.requestAnimationFrame : (cb) => cb();
  async function scroll_and_render(n) {
    raf(async () => {
      if (typeof n !== "number")
        return;
      const direction = typeof n !== "number" ? false : is_in_view(n);
      if (direction === true) {
        return;
      }
      if (direction === "back") {
        await scroll_to_index(n, { behavior: "instant" });
      }
      if (direction === "forwards") {
        await scroll_to_index(n, { behavior: "instant" }, true);
      }
    });
  }
  function is_in_view(n) {
    const current = rows ;
    if (n < start) {
      return "back";
    }
    if (n >= end - 1) {
      return "forwards";
    }
    const { top: viewport_top } = viewport.getBoundingClientRect();
    const { top: top2, bottom: bottom2 } = current.getBoundingClientRect();
    if (top2 - viewport_top < 37) {
      return "back";
    }
    if (bottom2 - viewport_top > viewport_height) {
      return "forwards";
    }
    return true;
  }
  async function scroll_to_index(index, opts, align_end = false) {
    await tick();
    const _itemHeight = average_height;
    let distance = index * _itemHeight;
    if (align_end) {
      distance = distance - viewport_height + _itemHeight + head_height;
    }
    viewport.offsetHeight - viewport.clientHeight;
    const _opts = {
      top: distance,
      behavior: "smooth",
      ...opts
    };
    viewport.scrollTo(_opts);
  }
  if ($$props.items === void 0 && $$bindings.items && items !== void 0)
    $$bindings.items(items);
  if ($$props.max_height === void 0 && $$bindings.max_height && max_height !== void 0)
    $$bindings.max_height(max_height);
  if ($$props.actual_height === void 0 && $$bindings.actual_height && actual_height !== void 0)
    $$bindings.actual_height(actual_height);
  if ($$props.table_scrollbar_width === void 0 && $$bindings.table_scrollbar_width && table_scrollbar_width !== void 0)
    $$bindings.table_scrollbar_width(table_scrollbar_width);
  if ($$props.start === void 0 && $$bindings.start && start !== void 0)
    $$bindings.start(start);
  if ($$props.end === void 0 && $$bindings.end && end !== void 0)
    $$bindings.end(end);
  if ($$props.selected === void 0 && $$bindings.selected && selected !== void 0)
    $$bindings.selected(selected);
  if ($$props.disable_scroll === void 0 && $$bindings.disable_scroll && disable_scroll !== void 0)
    $$bindings.disable_scroll(disable_scroll);
  if ($$props.scroll_to_index === void 0 && $$bindings.scroll_to_index && scroll_to_index !== void 0)
    $$bindings.scroll_to_index(scroll_to_index);
  $$result.css.add(css$4);
  viewport_height = 200;
  sortedItems = items;
  {
    scroll_and_render(selected);
  }
  visible = is_browser ? sortedItems.slice(start, end).map((data, i) => {
    return { index: i + start, data };
  }) : sortedItems.slice(0, max_height / sortedItems.length * average_height + 1).map((data, i) => {
    return { index: i + start, data };
  });
  return `<svelte-virtual-table-viewport><div><table class="${["table svelte-y11bhb", disable_scroll ? "disable-scroll" : ""].join(" ").trim()}" style="${"height: " + escape(height, true) + "; --bw-svt-p-top: " + escape(top, true) + "px; --bw-svt-p-bottom: " + escape(bottom, true) + "px; --bw-svt-head-height: " + escape(head_height, true) + "px; --bw-svt-foot-height: " + escape(foot_height, true) + "px; --bw-svt-avg-row-height: " + escape(average_height, true) + "px; --max-height: " + escape(max_height, true) + "px"}"${add_attribute("this", viewport, 0)}><thead class="thead svelte-y11bhb">${slots.thead ? slots.thead({}) : ``}</thead> <tbody class="tbody svelte-y11bhb"${add_attribute("this", contents, 0)}>${visible.length && visible[0].data.length ? `${each(visible, (item) => {
    return `${slots.tbody ? slots.tbody({ item: item.data, index: item.index }) : `
							Missing Table Row
						`}`;
  })}` : ``}</tbody> <tfoot class="tfoot svelte-y11bhb">${slots.tfoot ? slots.tfoot({}) : ``}</tfoot></table></div> </svelte-virtual-table-viewport>`;
});
const FilterIcon = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  return `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 4h16v2.67l-6.67 6.67v8L9.33 19v-5.66L2.67 6.67V4h1.33z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>`;
});
const css$2 = {
  code: ".toolbar.svelte-b1nr0g{display:flex;align-items:center;gap:var(--size-2);flex:0 0 auto}.toolbar-buttons.svelte-b1nr0g{display:flex;gap:var(--size-1);flex-wrap:nowrap}.toolbar-button.svelte-b1nr0g{display:flex;align-items:center;justify-content:center;width:var(--size-6);height:var(--size-6);padding:var(--size-1);border:none;border-radius:var(--radius-sm);background:transparent;color:var(--body-text-color-subdued);cursor:pointer;transition:all 0.2s}.toolbar-button.svelte-b1nr0g:hover{background:var(--background-fill-secondary);color:var(--body-text-color)}.toolbar-button.svelte-b1nr0g svg{width:var(--size-4);height:var(--size-4)}.search-container.svelte-b1nr0g{position:relative}.search-input.svelte-b1nr0g{width:var(--size-full);height:var(--size-6);padding:var(--size-2);padding-right:var(--size-8);border:1px solid var(--border-color-primary);border-radius:var(--table-radius);font-size:var(--text-sm);color:var(--body-text-color);background:var(--background-fill-secondary);transition:all 0.2s ease}.search-input.svelte-b1nr0g:hover{border-color:var(--border-color-secondary);background:var(--background-fill-primary)}.search-input.svelte-b1nr0g:focus{outline:none;border-color:var(--color-accent);background:var(--background-fill-primary);box-shadow:0 0 0 1px var(--color-accent)}.check-button.svelte-b1nr0g{position:absolute;right:var(--size-1);top:50%;transform:translateY(-50%);background:var(--color-accent);color:white;border:none;width:var(--size-4);height:var(--size-4);border-radius:var(--radius-sm);display:flex;align-items:center;justify-content:center;padding:var(--size-1)}.check-button.svelte-b1nr0g svg{width:var(--size-3);height:var(--size-3)}.check-button.svelte-b1nr0g:hover{background:var(--color-accent-soft)}",
  map: '{"version":3,"file":"Toolbar.svelte","sources":["Toolbar.svelte"],"sourcesContent":["<script lang=\\"ts\\">import { Maximize, Minimize, Copy } from \\"@gradio/icons\\";\\nimport { onDestroy } from \\"svelte\\";\\nimport { createEventDispatcher } from \\"svelte\\";\\nimport FilterIcon from \\"./icons/FilterIcon.svelte\\";\\nexport let show_fullscreen_button = false;\\nexport let show_copy_button = false;\\nexport let show_search = \\"none\\";\\nexport let is_fullscreen = false;\\nexport let on_copy;\\nexport let on_commit_filter;\\nconst dispatch = createEventDispatcher();\\nlet copied = false;\\nlet timer;\\nexport let current_search_query = null;\\n$: dispatch(\\"search\\", current_search_query);\\nfunction copy_feedback() {\\n    copied = true;\\n    if (timer)\\n        clearTimeout(timer);\\n    timer = setTimeout(() => {\\n        copied = false;\\n    }, 2e3);\\n}\\nasync function handle_copy() {\\n    await on_copy();\\n    copy_feedback();\\n}\\nonDestroy(() => {\\n    if (timer)\\n        clearTimeout(timer);\\n});\\n<\/script>\\n\\n<div class=\\"toolbar\\" role=\\"toolbar\\" aria-label=\\"Table actions\\">\\n\\t<div class=\\"toolbar-buttons\\">\\n\\t\\t{#if show_search !== \\"none\\"}\\n\\t\\t\\t<div class=\\"search-container\\">\\n\\t\\t\\t\\t<input\\n\\t\\t\\t\\t\\ttype=\\"text\\"\\n\\t\\t\\t\\t\\tbind:value={current_search_query}\\n\\t\\t\\t\\t\\tplaceholder=\\"Search...\\"\\n\\t\\t\\t\\t\\tclass=\\"search-input\\"\\n\\t\\t\\t\\t/>\\n\\t\\t\\t\\t{#if current_search_query && show_search === \\"filter\\"}\\n\\t\\t\\t\\t\\t<button\\n\\t\\t\\t\\t\\t\\tclass=\\"toolbar-button check-button\\"\\n\\t\\t\\t\\t\\t\\ton:click={on_commit_filter}\\n\\t\\t\\t\\t\\t\\taria-label=\\"Apply filter and update dataframe values\\"\\n\\t\\t\\t\\t\\t\\ttitle=\\"Apply filter and update dataframe values\\"\\n\\t\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t\\t<FilterIcon />\\n\\t\\t\\t\\t\\t</button>\\n\\t\\t\\t\\t{/if}\\n\\t\\t\\t</div>\\n\\t\\t{/if}\\n\\t\\t{#if show_copy_button}\\n\\t\\t\\t<button\\n\\t\\t\\t\\tclass=\\"toolbar-button\\"\\n\\t\\t\\t\\ton:click={handle_copy}\\n\\t\\t\\t\\taria-label={copied ? \\"Copied to clipboard\\" : \\"Copy table data\\"}\\n\\t\\t\\t\\ttitle={copied ? \\"Copied to clipboard\\" : \\"Copy table data\\"}\\n\\t\\t\\t>\\n\\t\\t\\t\\t{#if copied}\\n\\t\\t\\t\\t\\t<FilterIcon />\\n\\t\\t\\t\\t{:else}\\n\\t\\t\\t\\t\\t<Copy />\\n\\t\\t\\t\\t{/if}\\n\\t\\t\\t</button>\\n\\t\\t{/if}\\n\\t\\t{#if show_fullscreen_button}\\n\\t\\t\\t<button\\n\\t\\t\\t\\tclass=\\"toolbar-button\\"\\n\\t\\t\\t\\ton:click\\n\\t\\t\\t\\taria-label={is_fullscreen ? \\"Exit fullscreen\\" : \\"Enter fullscreen\\"}\\n\\t\\t\\t\\ttitle={is_fullscreen ? \\"Exit fullscreen\\" : \\"Enter fullscreen\\"}\\n\\t\\t\\t>\\n\\t\\t\\t\\t{#if is_fullscreen}\\n\\t\\t\\t\\t\\t<Minimize />\\n\\t\\t\\t\\t{:else}\\n\\t\\t\\t\\t\\t<Maximize />\\n\\t\\t\\t\\t{/if}\\n\\t\\t\\t</button>\\n\\t\\t{/if}\\n\\t</div>\\n</div>\\n\\n<style>\\n\\t.toolbar {\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\tgap: var(--size-2);\\n\\t\\tflex: 0 0 auto;\\n\\t}\\n\\n\\t.toolbar-buttons {\\n\\t\\tdisplay: flex;\\n\\t\\tgap: var(--size-1);\\n\\t\\tflex-wrap: nowrap;\\n\\t}\\n\\n\\t.toolbar-button {\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\tjustify-content: center;\\n\\t\\twidth: var(--size-6);\\n\\t\\theight: var(--size-6);\\n\\t\\tpadding: var(--size-1);\\n\\t\\tborder: none;\\n\\t\\tborder-radius: var(--radius-sm);\\n\\t\\tbackground: transparent;\\n\\t\\tcolor: var(--body-text-color-subdued);\\n\\t\\tcursor: pointer;\\n\\t\\ttransition: all 0.2s;\\n\\t}\\n\\n\\t.toolbar-button:hover {\\n\\t\\tbackground: var(--background-fill-secondary);\\n\\t\\tcolor: var(--body-text-color);\\n\\t}\\n\\n\\t.toolbar-button :global(svg) {\\n\\t\\twidth: var(--size-4);\\n\\t\\theight: var(--size-4);\\n\\t}\\n\\n\\t.search-container {\\n\\t\\tposition: relative;\\n\\t}\\n\\n\\t.search-input {\\n\\t\\twidth: var(--size-full);\\n\\t\\theight: var(--size-6);\\n\\t\\tpadding: var(--size-2);\\n\\t\\tpadding-right: var(--size-8);\\n\\t\\tborder: 1px solid var(--border-color-primary);\\n\\t\\tborder-radius: var(--table-radius);\\n\\t\\tfont-size: var(--text-sm);\\n\\t\\tcolor: var(--body-text-color);\\n\\t\\tbackground: var(--background-fill-secondary);\\n\\t\\ttransition: all 0.2s ease;\\n\\t}\\n\\n\\t.search-input:hover {\\n\\t\\tborder-color: var(--border-color-secondary);\\n\\t\\tbackground: var(--background-fill-primary);\\n\\t}\\n\\n\\t.search-input:focus {\\n\\t\\toutline: none;\\n\\t\\tborder-color: var(--color-accent);\\n\\t\\tbackground: var(--background-fill-primary);\\n\\t\\tbox-shadow: 0 0 0 1px var(--color-accent);\\n\\t}\\n\\n\\t.check-button {\\n\\t\\tposition: absolute;\\n\\t\\tright: var(--size-1);\\n\\t\\ttop: 50%;\\n\\t\\ttransform: translateY(-50%);\\n\\t\\tbackground: var(--color-accent);\\n\\t\\tcolor: white;\\n\\t\\tborder: none;\\n\\t\\twidth: var(--size-4);\\n\\t\\theight: var(--size-4);\\n\\t\\tborder-radius: var(--radius-sm);\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\tjustify-content: center;\\n\\t\\tpadding: var(--size-1);\\n\\t}\\n\\n\\t.check-button :global(svg) {\\n\\t\\twidth: var(--size-3);\\n\\t\\theight: var(--size-3);\\n\\t}\\n\\n\\t.check-button:hover {\\n\\t\\tbackground: var(--color-accent-soft);\\n\\t}</style>\\n"],"names":[],"mappings":"AAuFC,sBAAS,CACR,OAAO,CAAE,IAAI,CACb,WAAW,CAAE,MAAM,CACnB,GAAG,CAAE,IAAI,QAAQ,CAAC,CAClB,IAAI,CAAE,CAAC,CAAC,CAAC,CAAC,IACX,CAEA,8BAAiB,CAChB,OAAO,CAAE,IAAI,CACb,GAAG,CAAE,IAAI,QAAQ,CAAC,CAClB,SAAS,CAAE,MACZ,CAEA,6BAAgB,CACf,OAAO,CAAE,IAAI,CACb,WAAW,CAAE,MAAM,CACnB,eAAe,CAAE,MAAM,CACvB,KAAK,CAAE,IAAI,QAAQ,CAAC,CACpB,MAAM,CAAE,IAAI,QAAQ,CAAC,CACrB,OAAO,CAAE,IAAI,QAAQ,CAAC,CACtB,MAAM,CAAE,IAAI,CACZ,aAAa,CAAE,IAAI,WAAW,CAAC,CAC/B,UAAU,CAAE,WAAW,CACvB,KAAK,CAAE,IAAI,yBAAyB,CAAC,CACrC,MAAM,CAAE,OAAO,CACf,UAAU,CAAE,GAAG,CAAC,IACjB,CAEA,6BAAe,MAAO,CACrB,UAAU,CAAE,IAAI,2BAA2B,CAAC,CAC5C,KAAK,CAAE,IAAI,iBAAiB,CAC7B,CAEA,6BAAe,CAAS,GAAK,CAC5B,KAAK,CAAE,IAAI,QAAQ,CAAC,CACpB,MAAM,CAAE,IAAI,QAAQ,CACrB,CAEA,+BAAkB,CACjB,QAAQ,CAAE,QACX,CAEA,2BAAc,CACb,KAAK,CAAE,IAAI,WAAW,CAAC,CACvB,MAAM,CAAE,IAAI,QAAQ,CAAC,CACrB,OAAO,CAAE,IAAI,QAAQ,CAAC,CACtB,aAAa,CAAE,IAAI,QAAQ,CAAC,CAC5B,MAAM,CAAE,GAAG,CAAC,KAAK,CAAC,IAAI,sBAAsB,CAAC,CAC7C,aAAa,CAAE,IAAI,cAAc,CAAC,CAClC,SAAS,CAAE,IAAI,SAAS,CAAC,CACzB,KAAK,CAAE,IAAI,iBAAiB,CAAC,CAC7B,UAAU,CAAE,IAAI,2BAA2B,CAAC,CAC5C,UAAU,CAAE,GAAG,CAAC,IAAI,CAAC,IACtB,CAEA,2BAAa,MAAO,CACnB,YAAY,CAAE,IAAI,wBAAwB,CAAC,CAC3C,UAAU,CAAE,IAAI,yBAAyB,CAC1C,CAEA,2BAAa,MAAO,CACnB,OAAO,CAAE,IAAI,CACb,YAAY,CAAE,IAAI,cAAc,CAAC,CACjC,UAAU,CAAE,IAAI,yBAAyB,CAAC,CAC1C,UAAU,CAAE,CAAC,CAAC,CAAC,CAAC,CAAC,CAAC,GAAG,CAAC,IAAI,cAAc,CACzC,CAEA,2BAAc,CACb,QAAQ,CAAE,QAAQ,CAClB,KAAK,CAAE,IAAI,QAAQ,CAAC,CACpB,GAAG,CAAE,GAAG,CACR,SAAS,CAAE,WAAW,IAAI,CAAC,CAC3B,UAAU,CAAE,IAAI,cAAc,CAAC,CAC/B,KAAK,CAAE,KAAK,CACZ,MAAM,CAAE,IAAI,CACZ,KAAK,CAAE,IAAI,QAAQ,CAAC,CACpB,MAAM,CAAE,IAAI,QAAQ,CAAC,CACrB,aAAa,CAAE,IAAI,WAAW,CAAC,CAC/B,OAAO,CAAE,IAAI,CACb,WAAW,CAAE,MAAM,CACnB,eAAe,CAAE,MAAM,CACvB,OAAO,CAAE,IAAI,QAAQ,CACtB,CAEA,2BAAa,CAAS,GAAK,CAC1B,KAAK,CAAE,IAAI,QAAQ,CAAC,CACpB,MAAM,CAAE,IAAI,QAAQ,CACrB,CAEA,2BAAa,MAAO,CACnB,UAAU,CAAE,IAAI,mBAAmB,CACpC"}'
};
const Toolbar = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { show_fullscreen_button = false } = $$props;
  let { show_copy_button = false } = $$props;
  let { show_search = "none" } = $$props;
  let { is_fullscreen = false } = $$props;
  let { on_copy } = $$props;
  let { on_commit_filter } = $$props;
  const dispatch = createEventDispatcher();
  let { current_search_query = null } = $$props;
  onDestroy(() => {
  });
  if ($$props.show_fullscreen_button === void 0 && $$bindings.show_fullscreen_button && show_fullscreen_button !== void 0)
    $$bindings.show_fullscreen_button(show_fullscreen_button);
  if ($$props.show_copy_button === void 0 && $$bindings.show_copy_button && show_copy_button !== void 0)
    $$bindings.show_copy_button(show_copy_button);
  if ($$props.show_search === void 0 && $$bindings.show_search && show_search !== void 0)
    $$bindings.show_search(show_search);
  if ($$props.is_fullscreen === void 0 && $$bindings.is_fullscreen && is_fullscreen !== void 0)
    $$bindings.is_fullscreen(is_fullscreen);
  if ($$props.on_copy === void 0 && $$bindings.on_copy && on_copy !== void 0)
    $$bindings.on_copy(on_copy);
  if ($$props.on_commit_filter === void 0 && $$bindings.on_commit_filter && on_commit_filter !== void 0)
    $$bindings.on_commit_filter(on_commit_filter);
  if ($$props.current_search_query === void 0 && $$bindings.current_search_query && current_search_query !== void 0)
    $$bindings.current_search_query(current_search_query);
  $$result.css.add(css$2);
  {
    dispatch("search", current_search_query);
  }
  return `<div class="toolbar svelte-b1nr0g" role="toolbar" aria-label="Table actions"><div class="toolbar-buttons svelte-b1nr0g">${show_search !== "none" ? `<div class="search-container svelte-b1nr0g"><input type="text" placeholder="Search..." class="search-input svelte-b1nr0g"${add_attribute("value", current_search_query, 0)}> ${current_search_query && show_search === "filter" ? `<button class="toolbar-button check-button svelte-b1nr0g" aria-label="Apply filter and update dataframe values" title="Apply filter and update dataframe values">${validate_component(FilterIcon, "FilterIcon").$$render($$result, {}, {}, {})}</button>` : ``}</div>` : ``} ${show_copy_button ? `<button class="toolbar-button svelte-b1nr0g"${add_attribute("aria-label", "Copy table data", 0)}${add_attribute("title", "Copy table data", 0)}>${`${validate_component(Copy, "Copy").$$render($$result, {}, {}, {})}`}</button>` : ``} ${show_fullscreen_button ? `<button class="toolbar-button svelte-b1nr0g"${add_attribute("aria-label", is_fullscreen ? "Exit fullscreen" : "Enter fullscreen", 0)}${add_attribute("title", is_fullscreen ? "Exit fullscreen" : "Enter fullscreen", 0)}>${is_fullscreen ? `${validate_component(Minimize, "Minimize").$$render($$result, {}, {}, {})}` : `${validate_component(Maximize, "Maximize").$$render($$result, {}, {}, {})}`}</button>` : ``}</div> </div>`;
});
const css$1 = {
  code: ".sort-icons.svelte-a5uqm5{display:flex;flex-direction:column;gap:0;margin-right:var(--spacing-md)}.sort-button.svelte-a5uqm5{display:flex;align-items:center;justify-content:center;padding:0;background:none;border:none;cursor:pointer;opacity:0.5;transition:opacity 150ms}.sort-button.svelte-a5uqm5:hover{opacity:0.8}.sort-button.active.svelte-a5uqm5{opacity:1;color:var(--color-accent)}svg.svelte-a5uqm5{width:var(--size-3);height:var(--size-3);display:block}",
  map: '{"version":3,"file":"SortIcon.svelte","sources":["SortIcon.svelte"],"sourcesContent":["<script lang=\\"ts\\">import { createEventDispatcher } from \\"svelte\\";\\nexport let direction = null;\\nexport let i18n;\\nconst dispatch = createEventDispatcher();\\n<\/script>\\n\\n<div class=\\"sort-icons\\" role=\\"group\\" aria-label={i18n(\\"dataframe.sort_column\\")}>\\n\\t<button\\n\\t\\tclass=\\"sort-button up\\"\\n\\t\\tclass:active={direction === \\"asc\\"}\\n\\t\\ton:click={() => dispatch(\\"sort\\", \\"asc\\")}\\n\\t\\taria-label={i18n(\\"dataframe.sort_ascending\\")}\\n\\t\\taria-pressed={direction === \\"asc\\"}\\n\\t>\\n\\t\\t<svg\\n\\t\\t\\tviewBox=\\"0 0 24 24\\"\\n\\t\\t\\tfill=\\"none\\"\\n\\t\\t\\txmlns=\\"http://www.w3.org/2000/svg\\"\\n\\t\\t\\taria-hidden=\\"true\\"\\n\\t\\t\\tfocusable=\\"false\\"\\n\\t\\t>\\n\\t\\t\\t<path\\n\\t\\t\\t\\td=\\"M7 14l5-5 5 5\\"\\n\\t\\t\\t\\tstroke=\\"currentColor\\"\\n\\t\\t\\t\\tstroke-width=\\"2\\"\\n\\t\\t\\t\\tstroke-linecap=\\"round\\"\\n\\t\\t\\t\\tstroke-linejoin=\\"round\\"\\n\\t\\t\\t/>\\n\\t\\t</svg>\\n\\t</button>\\n\\t<button\\n\\t\\tclass=\\"sort-button down\\"\\n\\t\\tclass:active={direction === \\"des\\"}\\n\\t\\ton:click={() => dispatch(\\"sort\\", \\"des\\")}\\n\\t\\taria-label={i18n(\\"dataframe.sort_descending\\")}\\n\\t\\taria-pressed={direction === \\"des\\"}\\n\\t>\\n\\t\\t<svg\\n\\t\\t\\tviewBox=\\"0 0 24 24\\"\\n\\t\\t\\tfill=\\"none\\"\\n\\t\\t\\txmlns=\\"http://www.w3.org/2000/svg\\"\\n\\t\\t\\taria-hidden=\\"true\\"\\n\\t\\t\\tfocusable=\\"false\\"\\n\\t\\t>\\n\\t\\t\\t<path\\n\\t\\t\\t\\td=\\"M7 10l5 5 5-5\\"\\n\\t\\t\\t\\tstroke=\\"currentColor\\"\\n\\t\\t\\t\\tstroke-width=\\"2\\"\\n\\t\\t\\t\\tstroke-linecap=\\"round\\"\\n\\t\\t\\t\\tstroke-linejoin=\\"round\\"\\n\\t\\t\\t/>\\n\\t\\t</svg>\\n\\t</button>\\n</div>\\n\\n<style>\\n\\t.sort-icons {\\n\\t\\tdisplay: flex;\\n\\t\\tflex-direction: column;\\n\\t\\tgap: 0;\\n\\t\\tmargin-right: var(--spacing-md);\\n\\t}\\n\\n\\t.sort-button {\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\tjustify-content: center;\\n\\t\\tpadding: 0;\\n\\t\\tbackground: none;\\n\\t\\tborder: none;\\n\\t\\tcursor: pointer;\\n\\t\\topacity: 0.5;\\n\\t\\ttransition: opacity 150ms;\\n\\t}\\n\\n\\t.sort-button:hover {\\n\\t\\topacity: 0.8;\\n\\t}\\n\\n\\t.sort-button.active {\\n\\t\\topacity: 1;\\n\\t\\tcolor: var(--color-accent);\\n\\t}\\n\\n\\tsvg {\\n\\t\\twidth: var(--size-3);\\n\\t\\theight: var(--size-3);\\n\\t\\tdisplay: block;\\n\\t}</style>\\n"],"names":[],"mappings":"AAwDC,yBAAY,CACX,OAAO,CAAE,IAAI,CACb,cAAc,CAAE,MAAM,CACtB,GAAG,CAAE,CAAC,CACN,YAAY,CAAE,IAAI,YAAY,CAC/B,CAEA,0BAAa,CACZ,OAAO,CAAE,IAAI,CACb,WAAW,CAAE,MAAM,CACnB,eAAe,CAAE,MAAM,CACvB,OAAO,CAAE,CAAC,CACV,UAAU,CAAE,IAAI,CAChB,MAAM,CAAE,IAAI,CACZ,MAAM,CAAE,OAAO,CACf,OAAO,CAAE,GAAG,CACZ,UAAU,CAAE,OAAO,CAAC,KACrB,CAEA,0BAAY,MAAO,CAClB,OAAO,CAAE,GACV,CAEA,YAAY,qBAAQ,CACnB,OAAO,CAAE,CAAC,CACV,KAAK,CAAE,IAAI,cAAc,CAC1B,CAEA,iBAAI,CACH,KAAK,CAAE,IAAI,QAAQ,CAAC,CACpB,MAAM,CAAE,IAAI,QAAQ,CAAC,CACrB,OAAO,CAAE,KACV"}'
};
const SortIcon = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { direction = null } = $$props;
  let { i18n } = $$props;
  createEventDispatcher();
  if ($$props.direction === void 0 && $$bindings.direction && direction !== void 0)
    $$bindings.direction(direction);
  if ($$props.i18n === void 0 && $$bindings.i18n && i18n !== void 0)
    $$bindings.i18n(i18n);
  $$result.css.add(css$1);
  return `<div class="sort-icons svelte-a5uqm5" role="group"${add_attribute("aria-label", i18n("dataframe.sort_column"), 0)}><button class="${["sort-button up svelte-a5uqm5", direction === "asc" ? "active" : ""].join(" ").trim()}"${add_attribute("aria-label", i18n("dataframe.sort_ascending"), 0)}${add_attribute("aria-pressed", direction === "asc", 0)}><svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false" class="svelte-a5uqm5"><path d="M7 14l5-5 5 5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg></button> <button class="${["sort-button down svelte-a5uqm5", direction === "des" ? "active" : ""].join(" ").trim()}"${add_attribute("aria-label", i18n("dataframe.sort_descending"), 0)}${add_attribute("aria-pressed", direction === "des", 0)}><svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false" class="svelte-a5uqm5"><path d="M7 10l5 5 5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg></button> </div>`;
});
function is_cell_selected(cell, selected_cells) {
  const [row, col] = cell;
  if (!selected_cells.some(([r, c]) => r === row && c === col))
    return "";
  const up = selected_cells.some(([r, c]) => r === row - 1 && c === col);
  const down = selected_cells.some(([r, c]) => r === row + 1 && c === col);
  const left = selected_cells.some(([r, c]) => r === row && c === col - 1);
  const right = selected_cells.some(([r, c]) => r === row && c === col + 1);
  return `cell-selected${up ? " no-top" : ""}${down ? " no-bottom" : ""}${left ? " no-left" : ""}${right ? " no-right" : ""}`;
}
function should_show_cell_menu(cell, selected_cells, editable) {
  const [row, col] = cell;
  return editable && selected_cells.length === 1 && selected_cells[0][0] === row && selected_cells[0][1] === col;
}
function calculate_selection_positions(selected, data, els, parent, table) {
  const [row, col] = selected;
  if (!data[row]?.[col]) {
    return { col_pos: "0px", row_pos: void 0 };
  }
  let offset = 0;
  for (let i = 0; i < col; i++) {
    offset += parseFloat(
      getComputedStyle(parent).getPropertyValue(`--cell-width-${i}`)
    );
  }
  const cell_id = data[row][col].id;
  const cell_el = els[cell_id]?.cell;
  if (!cell_el) {
    return { col_pos: "0px", row_pos: void 0 };
  }
  const cell_rect = cell_el.getBoundingClientRect();
  const table_rect = table.getBoundingClientRect();
  const col_pos = `${cell_rect.left - table_rect.left + cell_rect.width / 2}px`;
  const relative_top = cell_rect.top - table_rect.top;
  const row_pos = `${relative_top + cell_rect.height / 2}px`;
  return { col_pos, row_pos };
}
function get_max(data) {
  if (!data || !data.length)
    return [];
  let max = data[0].slice();
  for (let i = 0; i < data.length; i++) {
    for (let j = 0; j < data[i].length; j++) {
      if (`${max[j].value}`.length < `${data[i][j].value}`.length) {
        max[j] = data[i][j];
      }
    }
  }
  return max;
}
async function copy_table_data(data, selected_cells) {
  const csv = selected_cells.reduce(
    (acc, [row, col]) => {
      acc[row] = acc[row] || {};
      const value = String(data[row][col].value);
      acc[row][col] = value.includes(",") || value.includes('"') || value.includes("\n") ? `"${value.replace(/"/g, '""')}"` : value;
      return acc;
    },
    {}
  );
  const rows = Object.keys(csv).sort((a, b) => +a - +b);
  const cols = Object.keys(csv[rows[0]]).sort((a, b) => +a - +b);
  const text = rows.map((r) => cols.map((c) => csv[r][c] || "").join(",")).join("\n");
  try {
    await navigator.clipboard.writeText(text);
  } catch (err) {
    console.error("Copy failed:", err);
    throw new Error("Failed to copy to clipboard: " + err.message);
  }
}
const css = {
  code: ".label.svelte-v7qhxw p.svelte-v7qhxw.svelte-v7qhxw{position:relative;z-index:var(--layer-4);margin-bottom:var(--size-2);color:var(--block-label-text-color);font-size:var(--block-label-text-size)}.table-container.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{display:flex;flex-direction:column;gap:var(--size-2)}.table-wrap.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{position:relative;transition:150ms}.table-wrap.menu-open.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{overflow:hidden}.table-wrap.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw:focus-within{outline:none}.dragging.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{border-color:var(--color-accent)}.no-wrap.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{white-space:nowrap}table.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{position:absolute;opacity:0;transition:150ms;width:var(--size-full);table-layout:auto;color:var(--body-text-color);font-size:var(--input-text-size);line-height:var(--line-md);font-family:var(--font-mono);border-spacing:0;border-collapse:separate}.table-wrap.svelte-v7qhxw>button{border:1px solid var(--border-color-primary);border-radius:var(--table-radius);overflow:hidden}div.svelte-v7qhxw:not(.no-wrap) td.svelte-v7qhxw.svelte-v7qhxw{overflow-wrap:anywhere}div.no-wrap.svelte-v7qhxw td.svelte-v7qhxw.svelte-v7qhxw{overflow-x:hidden}table.fixed-layout.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{table-layout:fixed}thead.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{position:sticky;top:0;z-index:var(--layer-2);box-shadow:var(--shadow-drop)}tr.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{border-bottom:1px solid var(--border-color-primary);text-align:left}tr.svelte-v7qhxw>.svelte-v7qhxw+.svelte-v7qhxw{border-right-width:0px;border-left-width:1px;border-style:solid;border-color:var(--border-color-primary)}th.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw,td.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{--ring-color:transparent;position:relative;outline:none;box-shadow:inset 0 0 0 1px var(--ring-color);padding:0}th.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw:first-child{border-top-left-radius:var(--table-radius);border-bottom-left-radius:var(--table-radius)}th.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw:last-child{border-top-right-radius:var(--table-radius);border-bottom-right-radius:var(--table-radius)}th.focus.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw,td.focus.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{--ring-color:var(--color-accent);box-shadow:inset 0 0 0 2px var(--ring-color);z-index:var(--layer-1)}th.focus.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{z-index:var(--layer-2)}tr.svelte-v7qhxw:last-child td.svelte-v7qhxw.svelte-v7qhxw:first-child{border-bottom-left-radius:var(--table-radius)}tr.svelte-v7qhxw:last-child td.svelte-v7qhxw.svelte-v7qhxw:last-child{border-bottom-right-radius:var(--table-radius)}tr.svelte-v7qhxw th.svelte-v7qhxw.svelte-v7qhxw{background:var(--table-even-background-fill)}.sort-buttons.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{display:flex;align-items:center;flex-shrink:0;order:-1}.editing.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{background:var(--table-editing)}.cell-wrap.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{display:flex;align-items:center;justify-content:flex-start;outline:none;min-height:var(--size-9);position:relative;height:100%;padding:var(--size-2);box-sizing:border-box;margin:0;gap:var(--size-1);overflow:visible;min-width:0;border-radius:var(--table-radius)}.header-content.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{display:flex;align-items:center;overflow:hidden;flex-grow:1;min-width:0;white-space:normal;overflow-wrap:break-word;word-break:normal;height:100%;gap:var(--size-1)}.row_odd.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{background:var(--table-odd-background-fill)}.row_odd.focus.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{background:var(--background-fill-primary)}.cell-menu-button.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{flex-shrink:0;display:none;background-color:var(--block-background-fill);border:1px solid var(--border-color-primary);border-radius:var(--block-radius);width:var(--size-5);height:var(--size-5);min-width:var(--size-5);padding:0;margin-right:var(--spacing-sm);z-index:var(--layer-1);position:absolute;right:var(--size-1);top:50%;transform:translateY(-50%)}.cell-selected.svelte-v7qhxw .cell-menu-button.svelte-v7qhxw.svelte-v7qhxw,th.svelte-v7qhxw:hover .cell-menu-button.svelte-v7qhxw.svelte-v7qhxw{display:flex;align-items:center;justify-content:center}.header-row.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{display:flex;justify-content:flex-end;align-items:center;gap:var(--size-2);min-height:var(--size-6);flex-wrap:nowrap;width:100%}.label.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{flex:1 1 auto;margin-right:auto}.label.svelte-v7qhxw p.svelte-v7qhxw.svelte-v7qhxw{margin:0;color:var(--block-label-text-color);font-size:var(--block-label-text-size);line-height:var(--line-sm)}.toolbar.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{flex:0 0 auto}.row-number.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw,.row-number-header.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{text-align:center;background:var(--table-even-background-fill);font-size:var(--input-text-size);color:var(--body-text-color);padding:var(--size-1);min-width:var(--size-12);width:var(--size-12);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:var(--weight-semibold)}.row-number-header.svelte-v7qhxw .header-content.svelte-v7qhxw.svelte-v7qhxw{justify-content:space-between;padding:var(--size-1);height:var(--size-9);display:flex;align-items:center}.row-number-header.svelte-v7qhxw .sort-icons{margin-right:0}tbody > tr:nth-child(odd) .row-number.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{background:var(--table-odd-background-fill)}.cell-selected.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{--ring-color:var(--color-accent);box-shadow:inset 0 0 0 2px var(--ring-color);z-index:var(--layer-1);position:relative}.cell-selected.no-top.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{box-shadow:inset 2px 0 0 var(--ring-color),\n			inset -2px 0 0 var(--ring-color),\n			inset 0 -2px 0 var(--ring-color)}.cell-selected.no-bottom.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{box-shadow:inset 2px 0 0 var(--ring-color),\n			inset -2px 0 0 var(--ring-color),\n			inset 0 2px 0 var(--ring-color)}.cell-selected.no-left.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{box-shadow:inset 0 2px 0 var(--ring-color),\n			inset -2px 0 0 var(--ring-color),\n			inset 0 -2px 0 var(--ring-color)}.cell-selected.no-right.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{box-shadow:inset 0 2px 0 var(--ring-color),\n			inset 2px 0 0 var(--ring-color),\n			inset 0 -2px 0 var(--ring-color)}.cell-selected.no-top.no-left.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{box-shadow:inset -2px 0 0 var(--ring-color),\n			inset 0 -2px 0 var(--ring-color)}.cell-selected.no-top.no-right.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{box-shadow:inset 2px 0 0 var(--ring-color),\n			inset 0 -2px 0 var(--ring-color)}.cell-selected.no-bottom.no-left.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{box-shadow:inset -2px 0 0 var(--ring-color),\n			inset 0 2px 0 var(--ring-color)}.cell-selected.no-bottom.no-right.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{box-shadow:inset 2px 0 0 var(--ring-color),\n			inset 0 2px 0 var(--ring-color)}.cell-selected.no-top.no-bottom.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{box-shadow:inset 2px 0 0 var(--ring-color),\n			inset -2px 0 0 var(--ring-color)}.cell-selected.no-left.no-right.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{box-shadow:inset 0 2px 0 var(--ring-color),\n			inset 0 -2px 0 var(--ring-color)}.cell-selected.no-top.no-left.no-right.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{box-shadow:inset 0 -2px 0 var(--ring-color)}.cell-selected.no-bottom.no-left.no-right.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{box-shadow:inset 0 2px 0 var(--ring-color)}.cell-selected.no-left.no-top.no-bottom.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{box-shadow:inset -2px 0 0 var(--ring-color)}.cell-selected.no-right.no-top.no-bottom.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{box-shadow:inset 2px 0 0 var(--ring-color)}.cell-selected.no-top.no-bottom.no-left.no-right.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{box-shadow:none}.selection-button.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{position:absolute;display:flex;align-items:center;justify-content:center;background:var(--color-accent);color:white;border-radius:var(--radius-sm);z-index:var(--layer-4)}.selection-button-column.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{width:var(--size-3);height:var(--size-5);top:-10px;left:var(--selected-col-pos);transform:rotate(90deg)}.selection-button-row.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{width:var(--size-3);height:var(--size-5);left:-7px;top:calc(var(--selected-row-pos) - var(--size-5) / 2)}.table-wrap.svelte-v7qhxw:not(:focus-within) .selection-button.svelte-v7qhxw.svelte-v7qhxw{opacity:0;pointer-events:none}.flash.cell-selected.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{animation:svelte-v7qhxw-flash-color 700ms ease-out}@keyframes svelte-v7qhxw-flash-color{0%,30%{background:var(--color-accent-copied)}100%{background:transparent}}.frozen-column.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{position:sticky;z-index:var(--layer-2);border-right:1px solid var(--border-color-primary)}tr.svelte-v7qhxw:nth-child(odd) .frozen-column.svelte-v7qhxw.svelte-v7qhxw{background:var(--table-odd-background-fill)}tr.svelte-v7qhxw:nth-child(even) .frozen-column.svelte-v7qhxw.svelte-v7qhxw{background:var(--table-even-background-fill)}.always-frozen.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{z-index:var(--layer-3)}.add-row-container.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{margin-top:var(--size-2)}.add-row-button.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw{width:100%;padding:var(--size-1);background:transparent;border:1px dashed var(--border-color-primary);border-radius:var(--radius-sm);color:var(--body-text-color);cursor:pointer;transition:all 150ms}.add-row-button.svelte-v7qhxw.svelte-v7qhxw.svelte-v7qhxw:hover{background:var(--background-fill-secondary);border-style:solid}",
  map: '{"version":3,"file":"Table.svelte","sources":["Table.svelte"],"sourcesContent":["<script lang=\\"ts\\">import { afterUpdate, createEventDispatcher, tick, onMount } from \\"svelte\\";\\nimport { dequal } from \\"dequal/lite\\";\\nimport { Upload } from \\"@gradio/upload\\";\\nimport EditableCell from \\"./EditableCell.svelte\\";\\nimport {} from \\"@gradio/client\\";\\nimport VirtualTable from \\"./VirtualTable.svelte\\";\\nimport CellMenu from \\"./CellMenu.svelte\\";\\nimport Toolbar from \\"./Toolbar.svelte\\";\\nimport SortIcon from \\"./icons/SortIcon.svelte\\";\\nimport { is_cell_selected, handle_selection, handle_delete_key, should_show_cell_menu, get_next_cell_coordinates, get_range_selection, move_cursor, get_current_indices, handle_click_outside as handle_click_outside_util, select_column, select_row, calculate_selection_positions } from \\"./selection_utils\\";\\nimport { copy_table_data, get_max, handle_file_upload, sort_table_data } from \\"./utils/table_utils\\";\\nexport let datatype;\\nexport let label = null;\\nexport let show_label = true;\\nexport let headers = [];\\nexport let values = [];\\nexport let col_count;\\nexport let row_count;\\nexport let latex_delimiters;\\nexport let editable = true;\\nexport let wrap = false;\\nexport let root;\\nexport let i18n;\\nexport let max_height = 500;\\nexport let line_breaks = true;\\nexport let column_widths = [];\\nexport let show_row_numbers = false;\\nexport let upload;\\nexport let stream_handler;\\nexport let show_fullscreen_button = false;\\nexport let show_copy_button = false;\\nexport let value_is_output = false;\\nexport let max_chars = void 0;\\nexport let show_search = \\"none\\";\\nexport let pinned_columns = 0;\\nlet actual_pinned_columns = 0;\\n$: actual_pinned_columns = pinned_columns && data?.[0]?.length ? Math.min(pinned_columns, data[0].length) : 0;\\nlet selected_cells = [];\\n$: selected_cells = [...selected_cells];\\nlet selected = false;\\n$: selected = selected_cells.length > 0 ? selected_cells[selected_cells.length - 1] : false;\\nexport let display_value = null;\\nexport let styling = null;\\nlet t_rect;\\nlet els = {};\\nlet data_binding = {};\\nconst dispatch = createEventDispatcher();\\nlet editing = false;\\nlet clear_on_focus = false;\\nlet header_edit = false;\\nlet selected_header = false;\\nlet active_cell_menu = null;\\nlet active_header_menu = null;\\nlet is_fullscreen = false;\\nlet dragging = false;\\nlet copy_flash = false;\\nlet color_accent_copied;\\nonMount(() => {\\n    const color = getComputedStyle(document.documentElement).getPropertyValue(\\"--color-accent\\").trim();\\n    color_accent_copied = color + \\"40\\";\\n    document.documentElement.style.setProperty(\\"--color-accent-copied\\", color_accent_copied);\\n});\\nconst get_data_at = (row, col) => data?.[row]?.[col]?.value;\\nfunction make_id() {\\n    return Math.random().toString(36).substring(2, 15);\\n}\\nfunction make_headers(_head, col_count2, els2) {\\n    let _h = _head || [];\\n    if (col_count2[1] === \\"fixed\\" && _h.length < col_count2[0]) {\\n        const fill = Array(col_count2[0] - _h.length).fill(\\"\\").map((_, i) => `${i + _h.length}`);\\n        _h = _h.concat(fill);\\n    }\\n    if (!_h || _h.length === 0) {\\n        return Array(col_count2[0]).fill(0).map((_, i) => {\\n            const _id = make_id();\\n            els2[_id] = { cell: null, input: null };\\n            return { id: _id, value: JSON.stringify(i + 1) };\\n        });\\n    }\\n    return _h.map((h, i) => {\\n        const _id = make_id();\\n        els2[_id] = { cell: null, input: null };\\n        return { id: _id, value: h ?? \\"\\" };\\n    });\\n}\\nfunction process_data(_values) {\\n    const data_row_length = _values.length;\\n    if (data_row_length === 0)\\n        return [];\\n    return Array(row_count[1] === \\"fixed\\" ? row_count[0] : data_row_length).fill(0).map((_, i) => {\\n        return Array(col_count[1] === \\"fixed\\" ? col_count[0] : _values[0].length || headers.length).fill(0).map((_2, j) => {\\n            const id = make_id();\\n            els[id] = els[id] || { input: null, cell: null };\\n            const obj = { value: _values?.[i]?.[j] ?? \\"\\", id };\\n            data_binding[id] = obj;\\n            return obj;\\n        });\\n    });\\n}\\nlet _headers = make_headers(headers, col_count, els);\\nlet old_headers = headers;\\n$: {\\n    if (!dequal(headers, old_headers)) {\\n        _headers = make_headers(headers, col_count, els);\\n        old_headers = JSON.parse(JSON.stringify(headers));\\n    }\\n}\\nlet data = [[]];\\nlet old_val = void 0;\\n$: if (!dequal(values, old_val)) {\\n    data = process_data(values);\\n    old_val = JSON.parse(JSON.stringify(values));\\n}\\nlet previous_headers = _headers.map((h) => h.value);\\nlet previous_data = data.map((row) => row.map((cell) => String(cell.value)));\\nasync function trigger_change() {\\n    if (current_search_query)\\n        return;\\n    const current_headers = _headers.map((h) => h.value);\\n    const current_data = data.map((row) => row.map((cell) => String(cell.value)));\\n    if (!dequal(current_data, previous_data) || !dequal(current_headers, previous_headers)) {\\n        dispatch(\\"change\\", {\\n            data: data.map((row) => row.map((cell) => cell.value)),\\n            headers: _headers.map((h) => h.value),\\n            metadata: null\\n            // the metadata (display value, styling) cannot be changed by the user so we don\'t need to pass it up\\n        });\\n        if (!value_is_output) {\\n            dispatch(\\"input\\");\\n        }\\n        previous_data = current_data;\\n        previous_headers = current_headers;\\n    }\\n}\\nfunction get_sort_status(name, _sort, direction) {\\n    if (!_sort)\\n        return \\"none\\";\\n    if (headers[_sort] === name) {\\n        if (direction === \\"asc\\")\\n            return \\"ascending\\";\\n        if (direction === \\"des\\")\\n            return \\"descending\\";\\n    }\\n    return \\"none\\";\\n}\\nasync function handle_keydown(event) {\\n    if (selected_header !== false && header_edit === false) {\\n        switch (event.key) {\\n            case \\"ArrowDown\\":\\n                selected = [0, selected_header];\\n                selected_cells = [[0, selected_header]];\\n                selected_header = false;\\n                return;\\n            case \\"ArrowLeft\\":\\n                selected_header = selected_header > 0 ? selected_header - 1 : selected_header;\\n                return;\\n            case \\"ArrowRight\\":\\n                selected_header = selected_header < _headers.length - 1 ? selected_header + 1 : selected_header;\\n                return;\\n            case \\"Escape\\":\\n                event.preventDefault();\\n                selected_header = false;\\n                break;\\n            case \\"Enter\\":\\n                event.preventDefault();\\n                if (editable) {\\n                    header_edit = selected_header;\\n                }\\n                break;\\n        }\\n    }\\n    if (event.key === \\"Delete\\" || event.key === \\"Backspace\\") {\\n        if (!editable)\\n            return;\\n        if (editing) {\\n            const [row, col] = editing;\\n            const input_el = els[data[row][col].id].input;\\n            if (input_el && input_el.selectionStart !== input_el.selectionEnd) {\\n                return;\\n            }\\n            if (event.key === \\"Delete\\" && input_el?.selectionStart !== input_el?.value.length) {\\n                return;\\n            }\\n            if (event.key === \\"Backspace\\" && input_el?.selectionStart !== 0) {\\n                return;\\n            }\\n        }\\n        event.preventDefault();\\n        if (selected_cells.length > 0) {\\n            data = handle_delete_key(data, selected_cells);\\n            dispatch(\\"change\\", {\\n                data: data.map((row) => row.map((cell) => cell.value)),\\n                headers: _headers.map((h) => h.value),\\n                metadata: null\\n            });\\n            if (!value_is_output) {\\n                dispatch(\\"input\\");\\n            }\\n        }\\n        return;\\n    }\\n    if (event.key === \\"c\\" && (event.metaKey || event.ctrlKey)) {\\n        event.preventDefault();\\n        if (selected_cells.length > 0) {\\n            await handle_copy();\\n        }\\n        return;\\n    }\\n    if (!selected) {\\n        return;\\n    }\\n    const [i, j] = selected;\\n    switch (event.key) {\\n        case \\"ArrowRight\\":\\n        case \\"ArrowLeft\\":\\n        case \\"ArrowDown\\":\\n        case \\"ArrowUp\\":\\n            if (editing)\\n                break;\\n            event.preventDefault();\\n            const next_coords = move_cursor(event.key, [i, j], data);\\n            if (next_coords) {\\n                if (event.shiftKey) {\\n                    selected_cells = get_range_selection(selected_cells.length > 0 ? selected_cells[0] : [i, j], next_coords);\\n                    editing = false;\\n                }\\n                else {\\n                    selected_cells = [next_coords];\\n                    editing = false;\\n                }\\n                selected = next_coords;\\n            }\\n            else if (next_coords === false && event.key === \\"ArrowUp\\" && i === 0) {\\n                selected_header = j;\\n                selected = false;\\n                selected_cells = [];\\n                editing = false;\\n            }\\n            break;\\n        case \\"Escape\\":\\n            if (!editable)\\n                break;\\n            event.preventDefault();\\n            editing = false;\\n            break;\\n        case \\"Enter\\":\\n            event.preventDefault();\\n            if (editable) {\\n                if (event.shiftKey) {\\n                    add_row(i);\\n                    await tick();\\n                    selected = [i + 1, j];\\n                }\\n                else {\\n                    if (dequal(editing, [i, j])) {\\n                        const cell_id = data[i][j].id;\\n                        const input_el = els[cell_id].input;\\n                        if (input_el) {\\n                            data[i][j].value = input_el.value;\\n                        }\\n                        editing = false;\\n                        await tick();\\n                        selected = [i, j];\\n                    }\\n                    else {\\n                        editing = [i, j];\\n                        clear_on_focus = false;\\n                    }\\n                }\\n            }\\n            break;\\n        case \\"Tab\\":\\n            event.preventDefault();\\n            editing = false;\\n            const next_cell = get_next_cell_coordinates([i, j], data, event.shiftKey);\\n            if (next_cell) {\\n                selected_cells = [next_cell];\\n                selected = next_cell;\\n                if (editable) {\\n                    editing = next_cell;\\n                    clear_on_focus = false;\\n                }\\n            }\\n            break;\\n        default:\\n            if (!editable)\\n                break;\\n            if ((!editing || editing && dequal(editing, [i, j])) && event.key.length === 1) {\\n                clear_on_focus = true;\\n                editing = [i, j];\\n            }\\n    }\\n}\\nlet sort_direction;\\nlet sort_by;\\nfunction handle_sort(col, direction) {\\n    if (typeof sort_by !== \\"number\\" || sort_by !== col) {\\n        sort_direction = direction;\\n        sort_by = col;\\n    }\\n    else if (sort_by === col) {\\n        if (sort_direction === direction) {\\n            sort_direction = void 0;\\n            sort_by = void 0;\\n        }\\n        else {\\n            sort_direction = direction;\\n        }\\n    }\\n}\\nasync function edit_header(i, _select = false) {\\n    if (!editable || header_edit === i)\\n        return;\\n    selected = false;\\n    selected_cells = [];\\n    selected_header = i;\\n    header_edit = i;\\n}\\nfunction end_header_edit(event) {\\n    if (!editable)\\n        return;\\n    switch (event.detail.key) {\\n        case \\"Escape\\":\\n        case \\"Enter\\":\\n        case \\"Tab\\":\\n            event.preventDefault();\\n            selected = false;\\n            selected_header = header_edit;\\n            header_edit = false;\\n            parent.focus();\\n            break;\\n    }\\n}\\nasync function add_row(index) {\\n    parent.focus();\\n    if (row_count[1] !== \\"dynamic\\")\\n        return;\\n    const new_row = Array(data[0]?.length || headers.length).fill(0).map((_, i) => {\\n        const _id = make_id();\\n        els[_id] = { cell: null, input: null };\\n        return { id: _id, value: \\"\\" };\\n    });\\n    if (data.length === 0) {\\n        data = [new_row];\\n    }\\n    else if (index !== void 0 && index >= 0 && index <= data.length) {\\n        data.splice(index, 0, new_row);\\n    }\\n    else {\\n        data.push(new_row);\\n    }\\n    data = data;\\n    selected = [index !== void 0 ? index : data.length - 1, 0];\\n}\\n$: (data || _headers) && trigger_change();\\nasync function add_col(index) {\\n    parent.focus();\\n    if (col_count[1] !== \\"dynamic\\")\\n        return;\\n    const insert_index = index !== void 0 ? index : data[0].length;\\n    for (let i = 0; i < data.length; i++) {\\n        const _id = make_id();\\n        els[_id] = { cell: null, input: null };\\n        data[i].splice(insert_index, 0, { id: _id, value: \\"\\" });\\n    }\\n    headers.splice(insert_index, 0, `Header ${headers.length + 1}`);\\n    data = data;\\n    headers = headers;\\n    await tick();\\n    requestAnimationFrame(() => {\\n        edit_header(insert_index, true);\\n        const new_w = parent.querySelectorAll(\\"tbody\\")[1].offsetWidth;\\n        parent.querySelectorAll(\\"table\\")[1].scrollTo({ left: new_w });\\n    });\\n}\\nfunction handle_click_outside(event) {\\n    if (handle_click_outside_util(event, parent)) {\\n        editing = false;\\n        selected_cells = [];\\n        header_edit = false;\\n        selected_header = false;\\n        active_cell_menu = null;\\n        active_header_menu = null;\\n    }\\n}\\n$: max = get_max(data);\\n$: cells[0] && set_cell_widths();\\nlet cells = [];\\nlet parent;\\nlet table;\\nfunction set_cell_widths() {\\n    const widths = cells.map((el) => el?.clientWidth || 0);\\n    if (widths.length === 0)\\n        return;\\n    if (show_row_numbers) {\\n        parent.style.setProperty(`--cell-width-row-number`, `${widths[0]}px`);\\n    }\\n    const data_cells = show_row_numbers ? widths.slice(1) : widths;\\n    data_cells.forEach((width, i) => {\\n        if (!column_widths[i]) {\\n            parent.style.setProperty(`--cell-width-${i}`, `${width - scrollbar_width / data_cells.length}px`);\\n        }\\n    });\\n}\\nfunction get_cell_width(index) {\\n    return column_widths[index] || `var(--cell-width-${index})`;\\n}\\nlet table_height = values.slice(0, max_height / values.length * 37).length * 37 + 37;\\nlet scrollbar_width = 0;\\nfunction sort_data(_data, _display_value, _styling, col, dir) {\\n    let id = null;\\n    if (selected && selected[0] in _data && selected[1] in _data[selected[0]]) {\\n        id = _data[selected[0]][selected[1]].id;\\n    }\\n    if (typeof col !== \\"number\\" || !dir) {\\n        return;\\n    }\\n    sort_table_data(_data, _display_value, _styling, col, dir);\\n    data = data;\\n    if (id) {\\n        const [i, j] = get_current_indices(id, data);\\n        selected = [i, j];\\n    }\\n}\\n$: sort_data(data, display_value, styling, sort_by, sort_direction);\\n$: selected_index = !!selected && selected[0];\\nlet is_visible = false;\\nonMount(() => {\\n    const observer = new IntersectionObserver((entries) => {\\n        entries.forEach((entry) => {\\n            if (entry.isIntersecting && !is_visible) {\\n                set_cell_widths();\\n                data = data;\\n            }\\n            is_visible = entry.isIntersecting;\\n        });\\n    });\\n    observer.observe(parent);\\n    document.addEventListener(\\"click\\", handle_click_outside);\\n    window.addEventListener(\\"resize\\", handle_resize);\\n    document.addEventListener(\\"fullscreenchange\\", handle_fullscreen_change);\\n    return () => {\\n        observer.disconnect();\\n        document.removeEventListener(\\"click\\", handle_click_outside);\\n        window.removeEventListener(\\"resize\\", handle_resize);\\n        document.removeEventListener(\\"fullscreenchange\\", handle_fullscreen_change);\\n    };\\n});\\nfunction handle_cell_click(event, row, col) {\\n    if (event.target instanceof HTMLAnchorElement) {\\n        return;\\n    }\\n    event.preventDefault();\\n    event.stopPropagation();\\n    if (show_row_numbers && col === -1)\\n        return;\\n    clear_on_focus = false;\\n    active_cell_menu = null;\\n    active_header_menu = null;\\n    selected_header = false;\\n    header_edit = false;\\n    selected_cells = handle_selection([row, col], selected_cells, event);\\n    parent.focus();\\n    if (editable) {\\n        if (selected_cells.length === 1) {\\n            editing = [row, col];\\n            tick().then(() => {\\n                const input_el = els[data[row][col].id].input;\\n                if (input_el) {\\n                    input_el.focus();\\n                    input_el.selectionStart = input_el.selectionEnd = input_el.value.length;\\n                }\\n            });\\n        }\\n        else {\\n            editing = false;\\n        }\\n    }\\n    toggle_cell_button(row, col);\\n    dispatch(\\"select\\", {\\n        index: [row, col],\\n        value: get_data_at(row, col),\\n        row_value: data[row].map((d) => d.value)\\n    });\\n}\\nfunction toggle_cell_menu(event, row, col) {\\n    event.stopPropagation();\\n    if (active_cell_menu && active_cell_menu.row === row && active_cell_menu.col === col) {\\n        active_cell_menu = null;\\n    }\\n    else {\\n        const cell = event.target.closest(\\"td\\");\\n        if (cell) {\\n            const rect = cell.getBoundingClientRect();\\n            active_cell_menu = { row, col, x: rect.right, y: rect.bottom };\\n        }\\n    }\\n}\\nfunction add_row_at(index, position) {\\n    const row_index = position === \\"above\\" ? index : index + 1;\\n    add_row(row_index);\\n    active_cell_menu = null;\\n    active_header_menu = null;\\n}\\nfunction add_col_at(index, position) {\\n    const col_index = position === \\"left\\" ? index : index + 1;\\n    add_col(col_index);\\n    active_cell_menu = null;\\n    active_header_menu = null;\\n}\\nfunction handle_resize() {\\n    active_cell_menu = null;\\n    active_header_menu = null;\\n    selected_cells = [];\\n    selected = false;\\n    editing = false;\\n    set_cell_widths();\\n}\\nlet active_button = null;\\nfunction toggle_header_button(col) {\\n    active_button = active_button?.type === \\"header\\" && active_button.col === col ? null : { type: \\"header\\", col };\\n}\\nfunction toggle_cell_button(row, col) {\\n    active_button = active_button?.type === \\"cell\\" && active_button.row === row && active_button.col === col ? null : { type: \\"cell\\", row, col };\\n}\\nfunction toggle_fullscreen() {\\n    if (!document.fullscreenElement) {\\n        parent.requestFullscreen();\\n        is_fullscreen = true;\\n    }\\n    else {\\n        document.exitFullscreen();\\n        is_fullscreen = false;\\n    }\\n}\\nfunction handle_fullscreen_change() {\\n    is_fullscreen = !!document.fullscreenElement;\\n}\\nasync function handle_copy() {\\n    await copy_table_data(data, selected_cells);\\n    copy_flash = true;\\n    setTimeout(() => {\\n        copy_flash = false;\\n    }, 800);\\n}\\nfunction toggle_header_menu(event, col) {\\n    event.stopPropagation();\\n    if (active_header_menu && active_header_menu.col === col) {\\n        active_header_menu = null;\\n    }\\n    else {\\n        const header = event.target.closest(\\"th\\");\\n        if (header) {\\n            const rect = header.getBoundingClientRect();\\n            active_header_menu = { col, x: rect.right, y: rect.bottom };\\n        }\\n    }\\n}\\nafterUpdate(() => {\\n    value_is_output = false;\\n});\\nasync function delete_row(index) {\\n    parent.focus();\\n    if (row_count[1] !== \\"dynamic\\")\\n        return;\\n    if (data.length <= 1)\\n        return;\\n    data.splice(index, 1);\\n    data = data;\\n    selected = false;\\n}\\nasync function delete_col(index) {\\n    parent.focus();\\n    if (col_count[1] !== \\"dynamic\\")\\n        return;\\n    if (_headers.length <= 1)\\n        return;\\n    _headers.splice(index, 1);\\n    _headers = _headers;\\n    if (data.length > 0) {\\n        data.forEach((row) => {\\n            row.splice(index, 1);\\n        });\\n        data = data;\\n    }\\n    selected = false;\\n}\\nfunction delete_row_at(index) {\\n    delete_row(index);\\n    active_cell_menu = null;\\n    active_header_menu = null;\\n}\\nfunction delete_col_at(index) {\\n    delete_col(index);\\n    active_cell_menu = null;\\n    active_header_menu = null;\\n}\\nlet row_order = [];\\n$: {\\n    if (typeof sort_by === \\"number\\" && sort_direction && sort_by >= 0 && sort_by < data[0].length) {\\n        const indices = [...Array(data.length)].map((_, i) => i);\\n        const sort_index = sort_by;\\n        indices.sort((a, b) => {\\n            const row_a = data[a];\\n            const row_b = data[b];\\n            if (!row_a || !row_b || sort_index >= row_a.length || sort_index >= row_b.length)\\n                return 0;\\n            const val_a = row_a[sort_index].value;\\n            const val_b = row_b[sort_index].value;\\n            const comp = val_a < val_b ? -1 : val_a > val_b ? 1 : 0;\\n            return sort_direction === \\"asc\\" ? comp : -comp;\\n        });\\n        row_order = indices;\\n    }\\n    else {\\n        row_order = [...Array(data.length)].map((_, i) => i);\\n    }\\n}\\nfunction handle_select_column(col) {\\n    selected_cells = select_column(data, col);\\n    selected = selected_cells[0];\\n    editing = false;\\n}\\nfunction handle_select_row(row) {\\n    selected_cells = select_row(data, row);\\n    selected = selected_cells[0];\\n    editing = false;\\n}\\nlet coords;\\n$: if (selected !== false)\\n    coords = selected;\\n$: if (selected !== false) {\\n    const positions = calculate_selection_positions(selected, data, els, parent, table);\\n    document.documentElement.style.setProperty(\\"--selected-col-pos\\", positions.col_pos);\\n    if (positions.row_pos) {\\n        document.documentElement.style.setProperty(\\"--selected-row-pos\\", positions.row_pos);\\n    }\\n}\\nlet current_search_query = null;\\nfunction handle_search(search_query) {\\n    current_search_query = search_query;\\n    dispatch(\\"search\\", search_query);\\n}\\nfunction commit_filter() {\\n    if (current_search_query && show_search === \\"filter\\") {\\n        dispatch(\\"change\\", {\\n            data: data.map((row) => row.map((cell) => cell.value)),\\n            headers: _headers.map((h) => h.value),\\n            metadata: null\\n        });\\n        if (!value_is_output) {\\n            dispatch(\\"input\\");\\n        }\\n        current_search_query = null;\\n    }\\n}\\nfunction handle_header_click(event, col) {\\n    if (event.target instanceof HTMLAnchorElement) {\\n        return;\\n    }\\n    event.preventDefault();\\n    event.stopPropagation();\\n    if (!editable)\\n        return;\\n    clear_on_focus = false;\\n    active_cell_menu = null;\\n    active_header_menu = null;\\n    selected = false;\\n    selected_cells = [];\\n    selected_header = col;\\n    header_edit = col;\\n    parent.focus();\\n}\\n<\/script>\\n\\n<svelte:window on:resize={() => set_cell_widths()} />\\n\\n<div class=\\"table-container\\">\\n\\t{#if (label && label.length !== 0 && show_label) || show_fullscreen_button || show_copy_button || show_search !== \\"none\\"}\\n\\t\\t<div class=\\"header-row\\">\\n\\t\\t\\t{#if label && label.length !== 0 && show_label}\\n\\t\\t\\t\\t<div class=\\"label\\">\\n\\t\\t\\t\\t\\t<p>{label}</p>\\n\\t\\t\\t\\t</div>\\n\\t\\t\\t{/if}\\n\\t\\t\\t<Toolbar\\n\\t\\t\\t\\t{show_fullscreen_button}\\n\\t\\t\\t\\t{is_fullscreen}\\n\\t\\t\\t\\ton:click={toggle_fullscreen}\\n\\t\\t\\t\\ton_copy={handle_copy}\\n\\t\\t\\t\\t{show_copy_button}\\n\\t\\t\\t\\t{show_search}\\n\\t\\t\\t\\ton:search={(e) => handle_search(e.detail)}\\n\\t\\t\\t\\ton_commit_filter={commit_filter}\\n\\t\\t\\t\\t{current_search_query}\\n\\t\\t\\t/>\\n\\t\\t</div>\\n\\t{/if}\\n\\t<div\\n\\t\\tbind:this={parent}\\n\\t\\tclass=\\"table-wrap\\"\\n\\t\\tclass:dragging\\n\\t\\tclass:no-wrap={!wrap}\\n\\t\\tstyle=\\"height:{table_height}px;\\"\\n\\t\\tclass:menu-open={active_cell_menu || active_header_menu}\\n\\t\\ton:keydown={(e) => handle_keydown(e)}\\n\\t\\trole=\\"grid\\"\\n\\t\\ttabindex=\\"0\\"\\n\\t>\\n\\t\\t{#if selected !== false && selected_cells.length === 1}\\n\\t\\t\\t<button\\n\\t\\t\\t\\tclass=\\"selection-button selection-button-column\\"\\n\\t\\t\\t\\ton:click|stopPropagation={() => handle_select_column(coords[1])}\\n\\t\\t\\t\\taria-label=\\"Select column\\"\\n\\t\\t\\t>\\n\\t\\t\\t\\t&#8942;\\n\\t\\t\\t</button>\\n\\t\\t\\t<button\\n\\t\\t\\t\\tclass=\\"selection-button selection-button-row\\"\\n\\t\\t\\t\\ton:click|stopPropagation={() => handle_select_row(coords[0])}\\n\\t\\t\\t\\taria-label=\\"Select row\\"\\n\\t\\t\\t>\\n\\t\\t\\t\\t&#8942;\\n\\t\\t\\t</button>\\n\\t\\t{/if}\\n\\t\\t<table\\n\\t\\t\\tbind:contentRect={t_rect}\\n\\t\\t\\tbind:this={table}\\n\\t\\t\\tclass:fixed-layout={column_widths.length != 0}\\n\\t\\t>\\n\\t\\t\\t{#if label && label.length !== 0}\\n\\t\\t\\t\\t<caption class=\\"sr-only\\">{label}</caption>\\n\\t\\t\\t{/if}\\n\\t\\t\\t<thead>\\n\\t\\t\\t\\t<tr>\\n\\t\\t\\t\\t\\t{#if show_row_numbers}\\n\\t\\t\\t\\t\\t\\t<th\\n\\t\\t\\t\\t\\t\\t\\tclass=\\"row-number-header frozen-column always-frozen\\"\\n\\t\\t\\t\\t\\t\\t\\tstyle=\\"left: 0;\\"\\n\\t\\t\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t\\t\\t<div class=\\"cell-wrap\\">\\n\\t\\t\\t\\t\\t\\t\\t\\t<div class=\\"header-content\\">\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t<div class=\\"header-text\\"></div>\\n\\t\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t</th>\\n\\t\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t\\t{#each _headers as { value, id }, i (id)}\\n\\t\\t\\t\\t\\t\\t<th\\n\\t\\t\\t\\t\\t\\t\\tclass:frozen-column={i < actual_pinned_columns}\\n\\t\\t\\t\\t\\t\\t\\tclass:last-frozen={i === actual_pinned_columns - 1}\\n\\t\\t\\t\\t\\t\\t\\tclass:focus={header_edit === i || selected_header === i}\\n\\t\\t\\t\\t\\t\\t\\taria-sort={get_sort_status(value, sort_by, sort_direction)}\\n\\t\\t\\t\\t\\t\\t\\tstyle=\\"width: {get_cell_width(i)}; left: {i <\\n\\t\\t\\t\\t\\t\\t\\tactual_pinned_columns\\n\\t\\t\\t\\t\\t\\t\\t\\t? i === 0\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t? show_row_numbers\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t? \'var(--cell-width-row-number)\'\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t: \'0\'\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t: `calc(${show_row_numbers ? \'var(--cell-width-row-number) + \' : \'\'}${Array(\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\ti\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t)\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t.fill(0)\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t.map((_, idx) => `var(--cell-width-${idx})`)\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t.join(\' + \')})`\\n\\t\\t\\t\\t\\t\\t\\t\\t: \'auto\'};\\"\\n\\t\\t\\t\\t\\t\\t\\ton:click={(event) => handle_header_click(event, i)}\\n\\t\\t\\t\\t\\t\\t\\ton:mousedown={(event) => {\\n\\t\\t\\t\\t\\t\\t\\t\\tevent.preventDefault();\\n\\t\\t\\t\\t\\t\\t\\t\\tevent.stopPropagation();\\n\\t\\t\\t\\t\\t\\t\\t}}\\n\\t\\t\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t\\t\\t<div class=\\"cell-wrap\\">\\n\\t\\t\\t\\t\\t\\t\\t\\t<div class=\\"header-content\\">\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t<EditableCell\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{max_chars}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tbind:value={_headers[i].value}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tbind:el={els[id].input}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{latex_delimiters}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{line_breaks}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tedit={header_edit === i}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\ton:keydown={end_header_edit}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\theader\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{root}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{editable}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t/>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{#if header_edit !== i}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t<div class=\\"sort-buttons\\">\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t<SortIcon\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tdirection={sort_by === i ? sort_direction : null}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\ton:sort={({ detail }) => handle_sort(i, detail)}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{i18n}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t/>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t\\t\\t{#if editable}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t<button\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tclass=\\"cell-menu-button\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\ton:click={(event) => toggle_header_menu(event, i)}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\ton:touchstart={(event) => {\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tevent.preventDefault();\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tconst touch = event.touches[0];\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tconst mouseEvent = new MouseEvent(\\"click\\", {\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tclientX: touch.clientX,\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tclientY: touch.clientY,\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tbubbles: true,\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tcancelable: true,\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tview: window\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t});\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\ttoggle_header_menu(mouseEvent, i);\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t}}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t&#8942;\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t</button>\\n\\t\\t\\t\\t\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t</th>\\n\\t\\t\\t\\t\\t{/each}\\n\\t\\t\\t\\t</tr>\\n\\t\\t\\t</thead>\\n\\t\\t\\t<tbody>\\n\\t\\t\\t\\t<tr>\\n\\t\\t\\t\\t\\t{#each max as { value, id }, j (id)}\\n\\t\\t\\t\\t\\t\\t<td tabindex=\\"-1\\" bind:this={cells[j]}>\\n\\t\\t\\t\\t\\t\\t\\t<div class=\\"cell-wrap\\">\\n\\t\\t\\t\\t\\t\\t\\t\\t<EditableCell\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{value}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{latex_delimiters}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{line_breaks}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\tdatatype={Array.isArray(datatype) ? datatype[j] : datatype}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\tedit={false}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\tel={null}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{root}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{editable}\\n\\t\\t\\t\\t\\t\\t\\t\\t/>\\n\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t</td>\\n\\t\\t\\t\\t\\t{/each}\\n\\t\\t\\t\\t</tr>\\n\\t\\t\\t</tbody>\\n\\t\\t</table>\\n\\t\\t<Upload\\n\\t\\t\\t{upload}\\n\\t\\t\\t{stream_handler}\\n\\t\\t\\tflex={false}\\n\\t\\t\\tcenter={false}\\n\\t\\t\\tboundedheight={false}\\n\\t\\t\\tdisable_click={true}\\n\\t\\t\\t{root}\\n\\t\\t\\ton:load={({ detail }) =>\\n\\t\\t\\t\\thandle_file_upload(\\n\\t\\t\\t\\t\\tdetail.data,\\n\\t\\t\\t\\t\\t(head) => {\\n\\t\\t\\t\\t\\t\\t_headers = make_headers(\\n\\t\\t\\t\\t\\t\\t\\thead.map((h) => h ?? \\"\\"),\\n\\t\\t\\t\\t\\t\\t\\tcol_count,\\n\\t\\t\\t\\t\\t\\t\\tels\\n\\t\\t\\t\\t\\t\\t);\\n\\t\\t\\t\\t\\t\\treturn _headers;\\n\\t\\t\\t\\t\\t},\\n\\t\\t\\t\\t\\t(vals) => {\\n\\t\\t\\t\\t\\t\\tvalues = vals;\\n\\t\\t\\t\\t\\t}\\n\\t\\t\\t\\t)}\\n\\t\\t\\tbind:dragging\\n\\t\\t\\taria_label={i18n(\\"dataframe.drop_to_upload\\")}\\n\\t\\t>\\n\\t\\t\\t<div class=\\"table-wrap\\">\\n\\t\\t\\t\\t<VirtualTable\\n\\t\\t\\t\\t\\tbind:items={data}\\n\\t\\t\\t\\t\\t{max_height}\\n\\t\\t\\t\\t\\tbind:actual_height={table_height}\\n\\t\\t\\t\\t\\tbind:table_scrollbar_width={scrollbar_width}\\n\\t\\t\\t\\t\\tselected={selected_index}\\n\\t\\t\\t\\t\\tdisable_scroll={active_cell_menu !== null ||\\n\\t\\t\\t\\t\\t\\tactive_header_menu !== null}\\n\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t{#if label && label.length !== 0}\\n\\t\\t\\t\\t\\t\\t<caption class=\\"sr-only\\">{label}</caption>\\n\\t\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t\\t<tr slot=\\"thead\\">\\n\\t\\t\\t\\t\\t\\t{#if show_row_numbers}\\n\\t\\t\\t\\t\\t\\t\\t<th\\n\\t\\t\\t\\t\\t\\t\\t\\tclass=\\"row-number-header frozen-column always-frozen\\"\\n\\t\\t\\t\\t\\t\\t\\t\\tstyle=\\"left: 0;\\"\\n\\t\\t\\t\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t\\t\\t\\t<div class=\\"cell-wrap\\">\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t<div class=\\"header-content\\">\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t<div class=\\"header-text\\"></div>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t\\t</th>\\n\\t\\t\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t\\t\\t{#each _headers as { value, id }, i (id)}\\n\\t\\t\\t\\t\\t\\t\\t<th\\n\\t\\t\\t\\t\\t\\t\\t\\tclass:frozen-column={i < actual_pinned_columns}\\n\\t\\t\\t\\t\\t\\t\\t\\tclass:last-frozen={i === actual_pinned_columns - 1}\\n\\t\\t\\t\\t\\t\\t\\t\\tclass:focus={header_edit === i || selected_header === i}\\n\\t\\t\\t\\t\\t\\t\\t\\taria-sort={get_sort_status(value, sort_by, sort_direction)}\\n\\t\\t\\t\\t\\t\\t\\t\\tstyle=\\"width: {get_cell_width(i)}; left: {i <\\n\\t\\t\\t\\t\\t\\t\\t\\tactual_pinned_columns\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t? i === 0\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t? show_row_numbers\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t? \'var(--cell-width-row-number)\'\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t: \'0\'\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t: `calc(${show_row_numbers ? \'var(--cell-width-row-number) + \' : \'\'}${Array(\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\ti\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t)\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t.fill(0)\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t.map((_, idx) => `var(--cell-width-${idx})`)\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t.join(\' + \')})`\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t: \'auto\'};\\"\\n\\t\\t\\t\\t\\t\\t\\t\\ton:click={(event) => handle_header_click(event, i)}\\n\\t\\t\\t\\t\\t\\t\\t\\ton:mousedown={(event) => {\\n\\t\\t\\t\\t\\t\\t\\t\\t\\tevent.preventDefault();\\n\\t\\t\\t\\t\\t\\t\\t\\t\\tevent.stopPropagation();\\n\\t\\t\\t\\t\\t\\t\\t\\t}}\\n\\t\\t\\t\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t\\t\\t\\t<div class=\\"cell-wrap\\">\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t<div class=\\"header-content\\">\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t<EditableCell\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{max_chars}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tbind:value={_headers[i].value}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tbind:el={els[id].input}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{latex_delimiters}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{line_breaks}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tedit={header_edit === i}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\theader\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{root}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{editable}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t/>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{#if header_edit !== i}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t<div class=\\"sort-buttons\\">\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t<SortIcon\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tdirection={sort_by === i ? sort_direction : null}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\ton:sort={({ detail }) => handle_sort(i, detail)}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{i18n}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t/>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{#if editable}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t<button\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tclass=\\"cell-menu-button\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\ton:click={(event) => toggle_header_menu(event, i)}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\ton:touchstart={(event) => {\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tevent.preventDefault();\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tconst touch = event.touches[0];\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tconst mouseEvent = new MouseEvent(\\"click\\", {\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tclientX: touch.clientX,\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tclientY: touch.clientY,\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tbubbles: true,\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tcancelable: true,\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tview: window\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t});\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\ttoggle_header_menu(mouseEvent, i);\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t}}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t&#8942;\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t</button>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t\\t</th>\\n\\t\\t\\t\\t\\t\\t{/each}\\n\\t\\t\\t\\t\\t</tr>\\n\\t\\t\\t\\t\\t<tr slot=\\"tbody\\" let:item let:index class:row_odd={index % 2 === 0}>\\n\\t\\t\\t\\t\\t\\t{#if show_row_numbers}\\n\\t\\t\\t\\t\\t\\t\\t<td\\n\\t\\t\\t\\t\\t\\t\\t\\tclass=\\"row-number frozen-column always-frozen\\"\\n\\t\\t\\t\\t\\t\\t\\t\\tstyle=\\"left: 0;\\"\\n\\t\\t\\t\\t\\t\\t\\t\\ttabindex=\\"-1\\"\\n\\t\\t\\t\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t\\t\\t\\t{index + 1}\\n\\t\\t\\t\\t\\t\\t\\t</td>\\n\\t\\t\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t\\t\\t{#each item as { value, id }, j (id)}\\n\\t\\t\\t\\t\\t\\t\\t<td\\n\\t\\t\\t\\t\\t\\t\\t\\tclass:frozen-column={j < actual_pinned_columns}\\n\\t\\t\\t\\t\\t\\t\\t\\tclass:last-frozen={j === actual_pinned_columns - 1}\\n\\t\\t\\t\\t\\t\\t\\t\\ttabindex={show_row_numbers && j === 0 ? -1 : 0}\\n\\t\\t\\t\\t\\t\\t\\t\\tbind:this={els[id].cell}\\n\\t\\t\\t\\t\\t\\t\\t\\ton:touchstart={(event) => {\\n\\t\\t\\t\\t\\t\\t\\t\\t\\tconst touch = event.touches[0];\\n\\t\\t\\t\\t\\t\\t\\t\\t\\tconst mouseEvent = new MouseEvent(\\"click\\", {\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tclientX: touch.clientX,\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tclientY: touch.clientY,\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tbubbles: true,\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tcancelable: true,\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tview: window\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t});\\n\\t\\t\\t\\t\\t\\t\\t\\t\\thandle_cell_click(mouseEvent, index, j);\\n\\t\\t\\t\\t\\t\\t\\t\\t}}\\n\\t\\t\\t\\t\\t\\t\\t\\ton:mousedown={(event) => {\\n\\t\\t\\t\\t\\t\\t\\t\\t\\tevent.preventDefault();\\n\\t\\t\\t\\t\\t\\t\\t\\t\\tevent.stopPropagation();\\n\\t\\t\\t\\t\\t\\t\\t\\t}}\\n\\t\\t\\t\\t\\t\\t\\t\\ton:click={(event) => handle_cell_click(event, index, j)}\\n\\t\\t\\t\\t\\t\\t\\t\\tstyle=\\"width: {get_cell_width(j)}; left: {j <\\n\\t\\t\\t\\t\\t\\t\\t\\tactual_pinned_columns\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t? j === 0\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t? show_row_numbers\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t? \'var(--cell-width-row-number)\'\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t: \'0\'\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t: `calc(${show_row_numbers ? \'var(--cell-width-row-number) + \' : \'\'}${Array(\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tj\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t)\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t.fill(0)\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t.map((_, idx) => `var(--cell-width-${idx})`)\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t.join(\' + \')})`\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t: \'auto\'}; {styling?.[index]?.[j] || \'\'}\\"\\n\\t\\t\\t\\t\\t\\t\\t\\tclass:flash={copy_flash &&\\n\\t\\t\\t\\t\\t\\t\\t\\t\\tis_cell_selected([index, j], selected_cells)}\\n\\t\\t\\t\\t\\t\\t\\t\\tclass={is_cell_selected([index, j], selected_cells)}\\n\\t\\t\\t\\t\\t\\t\\t\\tclass:menu-active={active_cell_menu &&\\n\\t\\t\\t\\t\\t\\t\\t\\t\\tactive_cell_menu.row === index &&\\n\\t\\t\\t\\t\\t\\t\\t\\t\\tactive_cell_menu.col === j}\\n\\t\\t\\t\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t\\t\\t\\t<div class=\\"cell-wrap\\">\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t<EditableCell\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tbind:value={data[index][j].value}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tbind:el={els[id].input}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tdisplay_value={display_value?.[index]?.[j]}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{latex_delimiters}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{line_breaks}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{editable}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tedit={dequal(editing, [index, j])}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tdatatype={Array.isArray(datatype) ? datatype[j] : datatype}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\ton:blur={() => {\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tclear_on_focus = false;\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tparent.focus();\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t}}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\ton:focus={() => {\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tconst row = index;\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tconst col = j;\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tif (\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t!selected_cells.some(([r, c]) => r === row && c === col)\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t) {\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tselected_cells = [[row, col]];\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t}}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{clear_on_focus}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{root}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{max_chars}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t/>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{#if editable && should_show_cell_menu([index, j], selected_cells, editable)}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t<button\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tclass=\\"cell-menu-button\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\ton:click={(event) => toggle_cell_menu(event, index, j)}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t&#8942;\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t</button>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t\\t</td>\\n\\t\\t\\t\\t\\t\\t{/each}\\n\\t\\t\\t\\t\\t</tr>\\n\\t\\t\\t\\t</VirtualTable>\\n\\t\\t\\t</div>\\n\\t\\t</Upload>\\n\\t</div>\\n</div>\\n{#if data.length === 0 && editable && row_count[1] === \\"dynamic\\"}\\n\\t<div class=\\"add-row-container\\">\\n\\t\\t<button class=\\"add-row-button\\" on:click={() => add_row()}>\\n\\t\\t\\t<span>+</span>\\n\\t\\t</button>\\n\\t</div>\\n{/if}\\n\\n{#if active_cell_menu}\\n\\t<CellMenu\\n\\t\\tx={active_cell_menu.x}\\n\\t\\ty={active_cell_menu.y}\\n\\t\\trow={active_cell_menu.row}\\n\\t\\t{col_count}\\n\\t\\t{row_count}\\n\\t\\ton_add_row_above={() => add_row_at(active_cell_menu?.row || 0, \\"above\\")}\\n\\t\\ton_add_row_below={() => add_row_at(active_cell_menu?.row || 0, \\"below\\")}\\n\\t\\ton_add_column_left={() => add_col_at(active_cell_menu?.col || 0, \\"left\\")}\\n\\t\\ton_add_column_right={() => add_col_at(active_cell_menu?.col || 0, \\"right\\")}\\n\\t\\ton_delete_row={() => delete_row_at(active_cell_menu?.row || 0)}\\n\\t\\ton_delete_col={() => delete_col_at(active_cell_menu?.col || 0)}\\n\\t\\tcan_delete_rows={data.length > 1}\\n\\t\\tcan_delete_cols={data[0].length > 1}\\n\\t\\t{i18n}\\n\\t/>\\n{/if}\\n\\n{#if active_header_menu !== null}\\n\\t<CellMenu\\n\\t\\t{i18n}\\n\\t\\tx={active_header_menu.x}\\n\\t\\ty={active_header_menu.y}\\n\\t\\trow={-1}\\n\\t\\t{col_count}\\n\\t\\t{row_count}\\n\\t\\ton_add_row_above={() => add_row_at(active_cell_menu?.row ?? -1, \\"above\\")}\\n\\t\\ton_add_row_below={() => add_row_at(active_cell_menu?.row ?? -1, \\"below\\")}\\n\\t\\ton_add_column_left={() => add_col_at(active_header_menu?.col ?? -1, \\"left\\")}\\n\\t\\ton_add_column_right={() =>\\n\\t\\t\\tadd_col_at(active_header_menu?.col ?? -1, \\"right\\")}\\n\\t\\ton_delete_row={() => delete_row_at(active_cell_menu?.row ?? -1)}\\n\\t\\ton_delete_col={() => delete_col_at(active_header_menu?.col ?? -1)}\\n\\t\\tcan_delete_rows={false}\\n\\t\\tcan_delete_cols={_headers.length > 1}\\n\\t/>\\n{/if}\\n\\n<style>\\n\\t.label p {\\n\\t\\tposition: relative;\\n\\t\\tz-index: var(--layer-4);\\n\\t\\tmargin-bottom: var(--size-2);\\n\\t\\tcolor: var(--block-label-text-color);\\n\\t\\tfont-size: var(--block-label-text-size);\\n\\t}\\n\\n\\t.table-container {\\n\\t\\tdisplay: flex;\\n\\t\\tflex-direction: column;\\n\\t\\tgap: var(--size-2);\\n\\t}\\n\\n\\t.table-wrap {\\n\\t\\tposition: relative;\\n\\t\\ttransition: 150ms;\\n\\t}\\n\\n\\t.table-wrap.menu-open {\\n\\t\\toverflow: hidden;\\n\\t}\\n\\n\\t.table-wrap:focus-within {\\n\\t\\toutline: none;\\n\\t}\\n\\n\\t.dragging {\\n\\t\\tborder-color: var(--color-accent);\\n\\t}\\n\\n\\t.no-wrap {\\n\\t\\twhite-space: nowrap;\\n\\t}\\n\\n\\ttable {\\n\\t\\tposition: absolute;\\n\\t\\topacity: 0;\\n\\t\\ttransition: 150ms;\\n\\t\\twidth: var(--size-full);\\n\\t\\ttable-layout: auto;\\n\\t\\tcolor: var(--body-text-color);\\n\\t\\tfont-size: var(--input-text-size);\\n\\t\\tline-height: var(--line-md);\\n\\t\\tfont-family: var(--font-mono);\\n\\t\\tborder-spacing: 0;\\n\\t\\tborder-collapse: separate;\\n\\t}\\n\\n\\t.table-wrap > :global(button) {\\n\\t\\tborder: 1px solid var(--border-color-primary);\\n\\t\\tborder-radius: var(--table-radius);\\n\\t\\toverflow: hidden;\\n\\t}\\n\\n\\tdiv:not(.no-wrap) td {\\n\\t\\toverflow-wrap: anywhere;\\n\\t}\\n\\n\\tdiv.no-wrap td {\\n\\t\\toverflow-x: hidden;\\n\\t}\\n\\n\\ttable.fixed-layout {\\n\\t\\ttable-layout: fixed;\\n\\t}\\n\\n\\tthead {\\n\\t\\tposition: sticky;\\n\\t\\ttop: 0;\\n\\t\\tz-index: var(--layer-2);\\n\\t\\tbox-shadow: var(--shadow-drop);\\n\\t}\\n\\n\\ttr {\\n\\t\\tborder-bottom: 1px solid var(--border-color-primary);\\n\\t\\ttext-align: left;\\n\\t}\\n\\n\\ttr > * + * {\\n\\t\\tborder-right-width: 0px;\\n\\t\\tborder-left-width: 1px;\\n\\t\\tborder-style: solid;\\n\\t\\tborder-color: var(--border-color-primary);\\n\\t}\\n\\n\\tth,\\n\\ttd {\\n\\t\\t--ring-color: transparent;\\n\\t\\tposition: relative;\\n\\t\\toutline: none;\\n\\t\\tbox-shadow: inset 0 0 0 1px var(--ring-color);\\n\\t\\tpadding: 0;\\n\\t}\\n\\n\\tth:first-child {\\n\\t\\tborder-top-left-radius: var(--table-radius);\\n\\t\\tborder-bottom-left-radius: var(--table-radius);\\n\\t}\\n\\n\\tth:last-child {\\n\\t\\tborder-top-right-radius: var(--table-radius);\\n\\t\\tborder-bottom-right-radius: var(--table-radius);\\n\\t}\\n\\n\\tth.focus,\\n\\ttd.focus {\\n\\t\\t--ring-color: var(--color-accent);\\n\\t\\tbox-shadow: inset 0 0 0 2px var(--ring-color);\\n\\t\\tz-index: var(--layer-1);\\n\\t}\\n\\n\\tth.focus {\\n\\t\\tz-index: var(--layer-2);\\n\\t}\\n\\n\\ttr:last-child td:first-child {\\n\\t\\tborder-bottom-left-radius: var(--table-radius);\\n\\t}\\n\\n\\ttr:last-child td:last-child {\\n\\t\\tborder-bottom-right-radius: var(--table-radius);\\n\\t}\\n\\n\\ttr th {\\n\\t\\tbackground: var(--table-even-background-fill);\\n\\t}\\n\\n\\t.sort-buttons {\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\tflex-shrink: 0;\\n\\t\\torder: -1;\\n\\t}\\n\\n\\t.editing {\\n\\t\\tbackground: var(--table-editing);\\n\\t}\\n\\n\\t.cell-wrap {\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\tjustify-content: flex-start;\\n\\t\\toutline: none;\\n\\t\\tmin-height: var(--size-9);\\n\\t\\tposition: relative;\\n\\t\\theight: 100%;\\n\\t\\tpadding: var(--size-2);\\n\\t\\tbox-sizing: border-box;\\n\\t\\tmargin: 0;\\n\\t\\tgap: var(--size-1);\\n\\t\\toverflow: visible;\\n\\t\\tmin-width: 0;\\n\\t\\tborder-radius: var(--table-radius);\\n\\t}\\n\\n\\t.header-content {\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\toverflow: hidden;\\n\\t\\tflex-grow: 1;\\n\\t\\tmin-width: 0;\\n\\t\\twhite-space: normal;\\n\\t\\toverflow-wrap: break-word;\\n\\t\\tword-break: normal;\\n\\t\\theight: 100%;\\n\\t\\tgap: var(--size-1);\\n\\t}\\n\\n\\t.row_odd {\\n\\t\\tbackground: var(--table-odd-background-fill);\\n\\t}\\n\\n\\t.row_odd.focus {\\n\\t\\tbackground: var(--background-fill-primary);\\n\\t}\\n\\n\\t.cell-menu-button {\\n\\t\\tflex-shrink: 0;\\n\\t\\tdisplay: none;\\n\\t\\tbackground-color: var(--block-background-fill);\\n\\t\\tborder: 1px solid var(--border-color-primary);\\n\\t\\tborder-radius: var(--block-radius);\\n\\t\\twidth: var(--size-5);\\n\\t\\theight: var(--size-5);\\n\\t\\tmin-width: var(--size-5);\\n\\t\\tpadding: 0;\\n\\t\\tmargin-right: var(--spacing-sm);\\n\\t\\tz-index: var(--layer-1);\\n\\t\\tposition: absolute;\\n\\t\\tright: var(--size-1);\\n\\t\\ttop: 50%;\\n\\t\\ttransform: translateY(-50%);\\n\\t}\\n\\n\\t.cell-selected .cell-menu-button,\\n\\tth:hover .cell-menu-button {\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\tjustify-content: center;\\n\\t}\\n\\n\\t.header-row {\\n\\t\\tdisplay: flex;\\n\\t\\tjustify-content: flex-end;\\n\\t\\talign-items: center;\\n\\t\\tgap: var(--size-2);\\n\\t\\tmin-height: var(--size-6);\\n\\t\\tflex-wrap: nowrap;\\n\\t\\twidth: 100%;\\n\\t}\\n\\n\\t.label {\\n\\t\\tflex: 1 1 auto;\\n\\t\\tmargin-right: auto;\\n\\t}\\n\\n\\t.label p {\\n\\t\\tmargin: 0;\\n\\t\\tcolor: var(--block-label-text-color);\\n\\t\\tfont-size: var(--block-label-text-size);\\n\\t\\tline-height: var(--line-sm);\\n\\t}\\n\\n\\t.toolbar {\\n\\t\\tflex: 0 0 auto;\\n\\t}\\n\\n\\t.row-number,\\n\\t.row-number-header {\\n\\t\\ttext-align: center;\\n\\t\\tbackground: var(--table-even-background-fill);\\n\\t\\tfont-size: var(--input-text-size);\\n\\t\\tcolor: var(--body-text-color);\\n\\t\\tpadding: var(--size-1);\\n\\t\\tmin-width: var(--size-12);\\n\\t\\twidth: var(--size-12);\\n\\t\\toverflow: hidden;\\n\\t\\ttext-overflow: ellipsis;\\n\\t\\twhite-space: nowrap;\\n\\t\\tfont-weight: var(--weight-semibold);\\n\\t}\\n\\n\\t.row-number-header .header-content {\\n\\t\\tjustify-content: space-between;\\n\\t\\tpadding: var(--size-1);\\n\\t\\theight: var(--size-9);\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t}\\n\\n\\t.row-number-header :global(.sort-icons) {\\n\\t\\tmargin-right: 0;\\n\\t}\\n\\n\\t:global(tbody > tr:nth-child(odd)) .row-number {\\n\\t\\tbackground: var(--table-odd-background-fill);\\n\\t}\\n\\n\\t.cell-selected {\\n\\t\\t--ring-color: var(--color-accent);\\n\\t\\tbox-shadow: inset 0 0 0 2px var(--ring-color);\\n\\t\\tz-index: var(--layer-1);\\n\\t\\tposition: relative;\\n\\t}\\n\\n\\t.cell-selected.no-top {\\n\\t\\tbox-shadow:\\n\\t\\t\\tinset 2px 0 0 var(--ring-color),\\n\\t\\t\\tinset -2px 0 0 var(--ring-color),\\n\\t\\t\\tinset 0 -2px 0 var(--ring-color);\\n\\t}\\n\\n\\t.cell-selected.no-bottom {\\n\\t\\tbox-shadow:\\n\\t\\t\\tinset 2px 0 0 var(--ring-color),\\n\\t\\t\\tinset -2px 0 0 var(--ring-color),\\n\\t\\t\\tinset 0 2px 0 var(--ring-color);\\n\\t}\\n\\n\\t.cell-selected.no-left {\\n\\t\\tbox-shadow:\\n\\t\\t\\tinset 0 2px 0 var(--ring-color),\\n\\t\\t\\tinset -2px 0 0 var(--ring-color),\\n\\t\\t\\tinset 0 -2px 0 var(--ring-color);\\n\\t}\\n\\n\\t.cell-selected.no-right {\\n\\t\\tbox-shadow:\\n\\t\\t\\tinset 0 2px 0 var(--ring-color),\\n\\t\\t\\tinset 2px 0 0 var(--ring-color),\\n\\t\\t\\tinset 0 -2px 0 var(--ring-color);\\n\\t}\\n\\n\\t.cell-selected.no-top.no-left {\\n\\t\\tbox-shadow:\\n\\t\\t\\tinset -2px 0 0 var(--ring-color),\\n\\t\\t\\tinset 0 -2px 0 var(--ring-color);\\n\\t}\\n\\n\\t.cell-selected.no-top.no-right {\\n\\t\\tbox-shadow:\\n\\t\\t\\tinset 2px 0 0 var(--ring-color),\\n\\t\\t\\tinset 0 -2px 0 var(--ring-color);\\n\\t}\\n\\n\\t.cell-selected.no-bottom.no-left {\\n\\t\\tbox-shadow:\\n\\t\\t\\tinset -2px 0 0 var(--ring-color),\\n\\t\\t\\tinset 0 2px 0 var(--ring-color);\\n\\t}\\n\\n\\t.cell-selected.no-bottom.no-right {\\n\\t\\tbox-shadow:\\n\\t\\t\\tinset 2px 0 0 var(--ring-color),\\n\\t\\t\\tinset 0 2px 0 var(--ring-color);\\n\\t}\\n\\n\\t.cell-selected.no-top.no-bottom {\\n\\t\\tbox-shadow:\\n\\t\\t\\tinset 2px 0 0 var(--ring-color),\\n\\t\\t\\tinset -2px 0 0 var(--ring-color);\\n\\t}\\n\\n\\t.cell-selected.no-left.no-right {\\n\\t\\tbox-shadow:\\n\\t\\t\\tinset 0 2px 0 var(--ring-color),\\n\\t\\t\\tinset 0 -2px 0 var(--ring-color);\\n\\t}\\n\\n\\t.cell-selected.no-top.no-left.no-right {\\n\\t\\tbox-shadow: inset 0 -2px 0 var(--ring-color);\\n\\t}\\n\\n\\t.cell-selected.no-bottom.no-left.no-right {\\n\\t\\tbox-shadow: inset 0 2px 0 var(--ring-color);\\n\\t}\\n\\n\\t.cell-selected.no-left.no-top.no-bottom {\\n\\t\\tbox-shadow: inset -2px 0 0 var(--ring-color);\\n\\t}\\n\\n\\t.cell-selected.no-right.no-top.no-bottom {\\n\\t\\tbox-shadow: inset 2px 0 0 var(--ring-color);\\n\\t}\\n\\n\\t.cell-selected.no-top.no-bottom.no-left.no-right {\\n\\t\\tbox-shadow: none;\\n\\t}\\n\\n\\t.selection-button {\\n\\t\\tposition: absolute;\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\tjustify-content: center;\\n\\t\\tbackground: var(--color-accent);\\n\\t\\tcolor: white;\\n\\t\\tborder-radius: var(--radius-sm);\\n\\t\\tz-index: var(--layer-4);\\n\\t}\\n\\n\\t.selection-button-column {\\n\\t\\twidth: var(--size-3);\\n\\t\\theight: var(--size-5);\\n\\t\\ttop: -10px;\\n\\t\\tleft: var(--selected-col-pos);\\n\\t\\ttransform: rotate(90deg);\\n\\t}\\n\\n\\t.selection-button-row {\\n\\t\\twidth: var(--size-3);\\n\\t\\theight: var(--size-5);\\n\\t\\tleft: -7px;\\n\\t\\ttop: calc(var(--selected-row-pos) - var(--size-5) / 2);\\n\\t}\\n\\n\\t.table-wrap:not(:focus-within) .selection-button {\\n\\t\\topacity: 0;\\n\\t\\tpointer-events: none;\\n\\t}\\n\\n\\t.flash.cell-selected {\\n\\t\\tanimation: flash-color 700ms ease-out;\\n\\t}\\n\\n\\t@keyframes flash-color {\\n\\t\\t0%,\\n\\t\\t30% {\\n\\t\\t\\tbackground: var(--color-accent-copied);\\n\\t\\t}\\n\\n\\t\\t100% {\\n\\t\\t\\tbackground: transparent;\\n\\t\\t}\\n\\t}\\n\\n\\t.frozen-column {\\n\\t\\tposition: sticky;\\n\\t\\tz-index: var(--layer-2);\\n\\t\\tborder-right: 1px solid var(--border-color-primary);\\n\\t}\\n\\n\\ttr:nth-child(odd) .frozen-column {\\n\\t\\tbackground: var(--table-odd-background-fill);\\n\\t}\\n\\n\\ttr:nth-child(even) .frozen-column {\\n\\t\\tbackground: var(--table-even-background-fill);\\n\\t}\\n\\n\\t.always-frozen {\\n\\t\\tz-index: var(--layer-3);\\n\\t}\\n\\n\\t.add-row-container {\\n\\t\\tmargin-top: var(--size-2);\\n\\t}\\n\\n\\t.add-row-button {\\n\\t\\twidth: 100%;\\n\\t\\tpadding: var(--size-1);\\n\\t\\tbackground: transparent;\\n\\t\\tborder: 1px dashed var(--border-color-primary);\\n\\t\\tborder-radius: var(--radius-sm);\\n\\t\\tcolor: var(--body-text-color);\\n\\t\\tcursor: pointer;\\n\\t\\ttransition: all 150ms;\\n\\t}\\n\\n\\t.add-row-button:hover {\\n\\t\\tbackground: var(--background-fill-secondary);\\n\\t\\tborder-style: solid;\\n\\t}</style>\\n"],"names":[],"mappings":"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"}'
};
function make_id() {
  return Math.random().toString(36).substring(2, 15);
}
function make_headers(_head, col_count2, els2) {
  let _h = _head || [];
  if (col_count2[1] === "fixed" && _h.length < col_count2[0]) {
    const fill = Array(col_count2[0] - _h.length).fill("").map((_, i) => `${i + _h.length}`);
    _h = _h.concat(fill);
  }
  if (!_h || _h.length === 0) {
    return Array(col_count2[0]).fill(0).map((_, i) => {
      const _id = make_id();
      els2[_id] = { cell: null, input: null };
      return { id: _id, value: JSON.stringify(i + 1) };
    });
  }
  return _h.map((h, i) => {
    const _id = make_id();
    els2[_id] = { cell: null, input: null };
    return { id: _id, value: h ?? "" };
  });
}
const Table = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let max;
  let selected_index;
  let { datatype } = $$props;
  let { label = null } = $$props;
  let { show_label = true } = $$props;
  let { headers = [] } = $$props;
  let { values = [] } = $$props;
  let { col_count } = $$props;
  let { row_count } = $$props;
  let { latex_delimiters } = $$props;
  let { editable = true } = $$props;
  let { wrap = false } = $$props;
  let { root } = $$props;
  let { i18n } = $$props;
  let { max_height = 500 } = $$props;
  let { line_breaks = true } = $$props;
  let { column_widths = [] } = $$props;
  let { show_row_numbers = false } = $$props;
  let { upload } = $$props;
  let { stream_handler } = $$props;
  let { show_fullscreen_button = false } = $$props;
  let { show_copy_button = false } = $$props;
  let { value_is_output = false } = $$props;
  let { max_chars = void 0 } = $$props;
  let { show_search = "none" } = $$props;
  let { pinned_columns = 0 } = $$props;
  let actual_pinned_columns = 0;
  let selected_cells = [];
  let selected = false;
  let { display_value = null } = $$props;
  let { styling = null } = $$props;
  let els = {};
  const dispatch = createEventDispatcher();
  let editing = false;
  let clear_on_focus = false;
  let header_edit = false;
  let selected_header = false;
  let active_header_menu = null;
  let is_fullscreen = false;
  let dragging = false;
  let copy_flash = false;
  function process_data(_values) {
    const data_row_length = _values.length;
    if (data_row_length === 0)
      return [];
    return Array(row_count[1] === "fixed" ? row_count[0] : data_row_length).fill(0).map((_, i) => {
      return Array(col_count[1] === "fixed" ? col_count[0] : _values[0].length || headers.length).fill(0).map((_2, j) => {
        const id = make_id();
        els[id] = els[id] || { input: null, cell: null };
        const obj = { value: _values?.[i]?.[j] ?? "", id };
        return obj;
      });
    });
  }
  let _headers = make_headers(headers, col_count, els);
  let old_headers = headers;
  let data = [[]];
  let old_val = void 0;
  let previous_headers = _headers.map((h) => h.value);
  let previous_data = data.map((row) => row.map((cell) => String(cell.value)));
  async function trigger_change() {
    const current_headers = _headers.map((h) => h.value);
    const current_data = data.map((row) => row.map((cell) => String(cell.value)));
    if (!dequal(current_data, previous_data) || !dequal(current_headers, previous_headers)) {
      dispatch("change", {
        data: data.map((row) => row.map((cell) => cell.value)),
        headers: _headers.map((h) => h.value),
        metadata: null
      });
      if (!value_is_output) {
        dispatch("input");
      }
      previous_data = current_data;
      previous_headers = current_headers;
    }
  }
  function get_sort_status(name, _sort, direction) {
    return "none";
  }
  let sort_direction;
  let sort_by;
  let cells = [];
  let parent;
  let table;
  function get_cell_width(index) {
    return column_widths[index] || `var(--cell-width-${index})`;
  }
  let table_height = values.slice(0, max_height / values.length * 37).length * 37 + 37;
  let scrollbar_width = 0;
  function sort_data(_data, _display_value, _styling, col, dir) {
    if (selected && selected[0] in _data && selected[1] in _data[selected[0]]) {
      _data[selected[0]][selected[1]].id;
    }
    {
      return;
    }
  }
  async function handle_copy() {
    await copy_table_data(data, selected_cells);
    copy_flash = true;
    setTimeout(
      () => {
        copy_flash = false;
      },
      800
    );
  }
  let current_search_query = null;
  function commit_filter() {
  }
  if ($$props.datatype === void 0 && $$bindings.datatype && datatype !== void 0)
    $$bindings.datatype(datatype);
  if ($$props.label === void 0 && $$bindings.label && label !== void 0)
    $$bindings.label(label);
  if ($$props.show_label === void 0 && $$bindings.show_label && show_label !== void 0)
    $$bindings.show_label(show_label);
  if ($$props.headers === void 0 && $$bindings.headers && headers !== void 0)
    $$bindings.headers(headers);
  if ($$props.values === void 0 && $$bindings.values && values !== void 0)
    $$bindings.values(values);
  if ($$props.col_count === void 0 && $$bindings.col_count && col_count !== void 0)
    $$bindings.col_count(col_count);
  if ($$props.row_count === void 0 && $$bindings.row_count && row_count !== void 0)
    $$bindings.row_count(row_count);
  if ($$props.latex_delimiters === void 0 && $$bindings.latex_delimiters && latex_delimiters !== void 0)
    $$bindings.latex_delimiters(latex_delimiters);
  if ($$props.editable === void 0 && $$bindings.editable && editable !== void 0)
    $$bindings.editable(editable);
  if ($$props.wrap === void 0 && $$bindings.wrap && wrap !== void 0)
    $$bindings.wrap(wrap);
  if ($$props.root === void 0 && $$bindings.root && root !== void 0)
    $$bindings.root(root);
  if ($$props.i18n === void 0 && $$bindings.i18n && i18n !== void 0)
    $$bindings.i18n(i18n);
  if ($$props.max_height === void 0 && $$bindings.max_height && max_height !== void 0)
    $$bindings.max_height(max_height);
  if ($$props.line_breaks === void 0 && $$bindings.line_breaks && line_breaks !== void 0)
    $$bindings.line_breaks(line_breaks);
  if ($$props.column_widths === void 0 && $$bindings.column_widths && column_widths !== void 0)
    $$bindings.column_widths(column_widths);
  if ($$props.show_row_numbers === void 0 && $$bindings.show_row_numbers && show_row_numbers !== void 0)
    $$bindings.show_row_numbers(show_row_numbers);
  if ($$props.upload === void 0 && $$bindings.upload && upload !== void 0)
    $$bindings.upload(upload);
  if ($$props.stream_handler === void 0 && $$bindings.stream_handler && stream_handler !== void 0)
    $$bindings.stream_handler(stream_handler);
  if ($$props.show_fullscreen_button === void 0 && $$bindings.show_fullscreen_button && show_fullscreen_button !== void 0)
    $$bindings.show_fullscreen_button(show_fullscreen_button);
  if ($$props.show_copy_button === void 0 && $$bindings.show_copy_button && show_copy_button !== void 0)
    $$bindings.show_copy_button(show_copy_button);
  if ($$props.value_is_output === void 0 && $$bindings.value_is_output && value_is_output !== void 0)
    $$bindings.value_is_output(value_is_output);
  if ($$props.max_chars === void 0 && $$bindings.max_chars && max_chars !== void 0)
    $$bindings.max_chars(max_chars);
  if ($$props.show_search === void 0 && $$bindings.show_search && show_search !== void 0)
    $$bindings.show_search(show_search);
  if ($$props.pinned_columns === void 0 && $$bindings.pinned_columns && pinned_columns !== void 0)
    $$bindings.pinned_columns(pinned_columns);
  if ($$props.display_value === void 0 && $$bindings.display_value && display_value !== void 0)
    $$bindings.display_value(display_value);
  if ($$props.styling === void 0 && $$bindings.styling && styling !== void 0)
    $$bindings.styling(styling);
  $$result.css.add(css);
  let $$settled;
  let $$rendered;
  let previous_head = $$result.head;
  do {
    $$settled = true;
    $$result.head = previous_head;
    {
      if (!dequal(values, old_val)) {
        data = process_data(values);
        old_val = JSON.parse(JSON.stringify(values));
      }
    }
    actual_pinned_columns = pinned_columns && data?.[0]?.length ? Math.min(pinned_columns, data[0].length) : 0;
    selected_cells = [...selected_cells];
    selected = selected_cells.length > 0 ? selected_cells[selected_cells.length - 1] : false;
    {
      {
        if (!dequal(headers, old_headers)) {
          _headers = make_headers(headers, col_count, els);
          old_headers = JSON.parse(JSON.stringify(headers));
        }
      }
    }
    (data || _headers) && trigger_change();
    max = get_max(data);
    {
      sort_data(data);
    }
    selected_index = !!selected && selected[0];
    {
      {
        {
          [...Array(data.length)].map((_, i) => i);
        }
      }
    }
    {
      if (selected !== false) {
        const positions = calculate_selection_positions(selected, data, els, parent, table);
        document.documentElement.style.setProperty("--selected-col-pos", positions.col_pos);
        if (positions.row_pos) {
          document.documentElement.style.setProperty("--selected-row-pos", positions.row_pos);
        }
      }
    }
    $$rendered = ` <div class="table-container svelte-v7qhxw">${label && label.length !== 0 && show_label || show_fullscreen_button || show_copy_button || show_search !== "none" ? `<div class="header-row svelte-v7qhxw">${label && label.length !== 0 && show_label ? `<div class="label svelte-v7qhxw"><p class="svelte-v7qhxw">${escape(label)}</p></div>` : ``} ${validate_component(Toolbar, "Toolbar").$$render(
      $$result,
      {
        show_fullscreen_button,
        is_fullscreen,
        on_copy: handle_copy,
        show_copy_button,
        show_search,
        on_commit_filter: commit_filter,
        current_search_query
      },
      {},
      {}
    )}</div>` : ``} <div class="${[
      "table-wrap svelte-v7qhxw",
      (dragging ? "dragging" : "") + " " + (!wrap ? "no-wrap" : "") + " " + ("")
    ].join(" ").trim()}" style="${"height:" + escape(table_height, true) + "px;"}" role="grid" tabindex="0"${add_attribute("this", parent, 0)}>${selected !== false && selected_cells.length === 1 ? `<button class="selection-button selection-button-column svelte-v7qhxw" aria-label="Select column" data-svelte-h="svelte-1sqo8ds">⋮</button> <button class="selection-button selection-button-row svelte-v7qhxw" aria-label="Select row" data-svelte-h="svelte-mm502l">⋮</button>` : ``} <table class="${["svelte-v7qhxw", column_widths.length != 0 ? "fixed-layout" : ""].join(" ").trim()}"${add_attribute("this", table, 0)}>${label && label.length !== 0 ? `<caption class="sr-only">${escape(label)}</caption>` : ``} <thead class="svelte-v7qhxw"><tr class="svelte-v7qhxw">${show_row_numbers ? `<th class="row-number-header frozen-column always-frozen svelte-v7qhxw" style="left: 0;" data-svelte-h="svelte-16yee2u"><div class="cell-wrap svelte-v7qhxw"><div class="header-content svelte-v7qhxw"><div class="header-text"></div></div></div></th>` : ``} ${each(_headers, ({ value, id }, i) => {
      return `<th${add_attribute("aria-sort", get_sort_status(), 0)} style="${"width: " + escape(get_cell_width(i), true) + "; left: " + escape(
        i < actual_pinned_columns ? i === 0 ? show_row_numbers ? "var(--cell-width-row-number)" : "0" : `calc(${show_row_numbers ? "var(--cell-width-row-number) + " : ""}${Array(i).fill(0).map((_, idx) => `var(--cell-width-${idx})`).join(" + ")})` : "auto",
        true
      ) + ";"}" class="${[
        "svelte-v7qhxw",
        (i < actual_pinned_columns ? "frozen-column" : "") + " " + (i === actual_pinned_columns - 1 ? "last-frozen" : "") + " " + (header_edit === i || selected_header === i ? "focus" : "")
      ].join(" ").trim()}"><div class="cell-wrap svelte-v7qhxw"><div class="header-content svelte-v7qhxw">${validate_component(EditableCell, "EditableCell").$$render(
        $$result,
        {
          max_chars,
          latex_delimiters,
          line_breaks,
          edit: header_edit === i,
          header: true,
          root,
          editable,
          value: _headers[i].value,
          el: els[id].input
        },
        {
          value: ($$value) => {
            _headers[i].value = $$value;
            $$settled = false;
          },
          el: ($$value) => {
            els[id].input = $$value;
            $$settled = false;
          }
        },
        {}
      )} ${header_edit !== i ? `<div class="sort-buttons svelte-v7qhxw">${validate_component(SortIcon, "SortIcon").$$render(
        $$result,
        {
          direction: sort_by === i ? sort_direction : null,
          i18n
        },
        {},
        {}
      )} </div>` : ``}</div> ${editable ? `<button class="cell-menu-button svelte-v7qhxw" data-svelte-h="svelte-1346esv">⋮
									</button>` : ``}</div> </th>`;
    })}</tr></thead> <tbody><tr class="svelte-v7qhxw">${each(max, ({ value, id }, j) => {
      return `<td tabindex="-1" class="svelte-v7qhxw"${add_attribute("this", cells[j], 0)}><div class="cell-wrap svelte-v7qhxw">${validate_component(EditableCell, "EditableCell").$$render(
        $$result,
        {
          value,
          latex_delimiters,
          line_breaks,
          datatype: Array.isArray(datatype) ? datatype[j] : datatype,
          edit: false,
          el: null,
          root,
          editable
        },
        {},
        {}
      )}</div> </td>`;
    })}</tr></tbody></table> ${validate_component(Upload, "Upload").$$render(
      $$result,
      {
        upload,
        stream_handler,
        flex: false,
        center: false,
        boundedheight: false,
        disable_click: true,
        root,
        aria_label: i18n("dataframe.drop_to_upload"),
        dragging
      },
      {
        dragging: ($$value) => {
          dragging = $$value;
          $$settled = false;
        }
      },
      {
        default: () => {
          return `<div class="table-wrap svelte-v7qhxw">${validate_component(VirtualTable, "VirtualTable").$$render(
            $$result,
            {
              max_height,
              selected: selected_index,
              disable_scroll: active_header_menu !== null,
              items: data,
              actual_height: table_height,
              table_scrollbar_width: scrollbar_width
            },
            {
              items: ($$value) => {
                data = $$value;
                $$settled = false;
              },
              actual_height: ($$value) => {
                table_height = $$value;
                $$settled = false;
              },
              table_scrollbar_width: ($$value) => {
                scrollbar_width = $$value;
                $$settled = false;
              }
            },
            {
              tbody: ({ index, item }) => {
                return `<tr slot="tbody" class="${["svelte-v7qhxw", index % 2 === 0 ? "row_odd" : ""].join(" ").trim()}">${show_row_numbers ? `<td class="row-number frozen-column always-frozen svelte-v7qhxw" style="left: 0;" tabindex="-1">${escape(index + 1)}</td>` : ``} ${each(item, ({ value, id }, j) => {
                  return `<td${add_attribute("tabindex", show_row_numbers && j === 0 ? -1 : 0, 0)} style="${"width: " + escape(get_cell_width(j), true) + "; left: " + escape(
                    j < actual_pinned_columns ? j === 0 ? show_row_numbers ? "var(--cell-width-row-number)" : "0" : `calc(${show_row_numbers ? "var(--cell-width-row-number) + " : ""}${Array(j).fill(0).map((_, idx) => `var(--cell-width-${idx})`).join(" + ")})` : "auto",
                    true
                  ) + "; " + escape(styling?.[index]?.[j] || "", true)}" class="${[
                    escape(null_to_empty(is_cell_selected([index, j], selected_cells)), true) + " svelte-v7qhxw",
                    (j < actual_pinned_columns ? "frozen-column" : "") + " " + (j === actual_pinned_columns - 1 ? "last-frozen" : "") + " " + (copy_flash && is_cell_selected([index, j], selected_cells) ? "flash" : "") + " " + ("")
                  ].join(" ").trim()}"${add_attribute("this", els[id].cell, 0)}><div class="cell-wrap svelte-v7qhxw">${validate_component(EditableCell, "EditableCell").$$render(
                    $$result,
                    {
                      display_value: display_value?.[index]?.[j],
                      latex_delimiters,
                      line_breaks,
                      editable,
                      edit: dequal(editing, [index, j]),
                      datatype: Array.isArray(datatype) ? datatype[j] : datatype,
                      clear_on_focus,
                      root,
                      max_chars,
                      value: data[index][j].value,
                      el: els[id].input
                    },
                    {
                      value: ($$value) => {
                        data[index][j].value = $$value;
                        $$settled = false;
                      },
                      el: ($$value) => {
                        els[id].input = $$value;
                        $$settled = false;
                      }
                    },
                    {}
                  )} ${editable && should_show_cell_menu([index, j], selected_cells, editable) ? `<button class="cell-menu-button svelte-v7qhxw" data-svelte-h="svelte-1klcg3o">⋮
										</button>` : ``}</div> </td>`;
                })}</tr>`;
              },
              thead: () => {
                return `<tr slot="thead" class="svelte-v7qhxw">${show_row_numbers ? `<th class="row-number-header frozen-column always-frozen svelte-v7qhxw" style="left: 0;" data-svelte-h="svelte-12b7zfr"><div class="cell-wrap svelte-v7qhxw"><div class="header-content svelte-v7qhxw"><div class="header-text"></div></div></div></th>` : ``} ${each(_headers, ({ value, id }, i) => {
                  return `<th${add_attribute("aria-sort", get_sort_status(), 0)} style="${"width: " + escape(get_cell_width(i), true) + "; left: " + escape(
                    i < actual_pinned_columns ? i === 0 ? show_row_numbers ? "var(--cell-width-row-number)" : "0" : `calc(${show_row_numbers ? "var(--cell-width-row-number) + " : ""}${Array(i).fill(0).map((_, idx) => `var(--cell-width-${idx})`).join(" + ")})` : "auto",
                    true
                  ) + ";"}" class="${[
                    "svelte-v7qhxw",
                    (i < actual_pinned_columns ? "frozen-column" : "") + " " + (i === actual_pinned_columns - 1 ? "last-frozen" : "") + " " + (header_edit === i || selected_header === i ? "focus" : "")
                  ].join(" ").trim()}"><div class="cell-wrap svelte-v7qhxw"><div class="header-content svelte-v7qhxw">${validate_component(EditableCell, "EditableCell").$$render(
                    $$result,
                    {
                      max_chars,
                      latex_delimiters,
                      line_breaks,
                      edit: header_edit === i,
                      header: true,
                      root,
                      editable,
                      value: _headers[i].value,
                      el: els[id].input
                    },
                    {
                      value: ($$value) => {
                        _headers[i].value = $$value;
                        $$settled = false;
                      },
                      el: ($$value) => {
                        els[id].input = $$value;
                        $$settled = false;
                      }
                    },
                    {}
                  )} ${header_edit !== i ? `<div class="sort-buttons svelte-v7qhxw">${validate_component(SortIcon, "SortIcon").$$render(
                    $$result,
                    {
                      direction: sort_by === i ? sort_direction : null,
                      i18n
                    },
                    {},
                    {}
                  )} </div>` : ``}</div> ${editable ? `<button class="cell-menu-button svelte-v7qhxw" data-svelte-h="svelte-1qfa9rg">⋮
										</button>` : ``}</div> </th>`;
                })}</tr>`;
              },
              default: () => {
                return `${label && label.length !== 0 ? `<caption class="sr-only">${escape(label)}</caption>` : ``}`;
              }
            }
          )}</div>`;
        }
      }
    )}</div></div> ${data.length === 0 && editable && row_count[1] === "dynamic" ? `<div class="add-row-container svelte-v7qhxw"><button class="add-row-button svelte-v7qhxw" data-svelte-h="svelte-ko5r28"><span>+</span></button></div>` : ``} ${``} ${``}`;
  } while (!$$settled);
  return $$rendered;
});
const Table$1 = Table;
const Index = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let filtered_cell_values;
  let _headers;
  let display_value;
  let styling;
  let { headers = [] } = $$props;
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { visible = true } = $$props;
  let { value = {
    data: [["", "", ""]],
    headers: ["1", "2", "3"],
    metadata: null
  } } = $$props;
  let { value_is_output = false } = $$props;
  let { col_count } = $$props;
  let { row_count } = $$props;
  let { label = null } = $$props;
  let { show_label = true } = $$props;
  let { wrap } = $$props;
  let { datatype } = $$props;
  let { scale = null } = $$props;
  let { min_width = void 0 } = $$props;
  let { root } = $$props;
  let { line_breaks = true } = $$props;
  let { column_widths = [] } = $$props;
  let { gradio } = $$props;
  let { latex_delimiters } = $$props;
  let { max_height = void 0 } = $$props;
  let { loading_status } = $$props;
  let { interactive } = $$props;
  let { show_fullscreen_button = false } = $$props;
  let { max_chars = void 0 } = $$props;
  let { show_copy_button = false } = $$props;
  let { show_row_numbers = false } = $$props;
  let { show_search = "none" } = $$props;
  let { pinned_columns = 0 } = $$props;
  if ($$props.headers === void 0 && $$bindings.headers && headers !== void 0)
    $$bindings.headers(headers);
  if ($$props.elem_id === void 0 && $$bindings.elem_id && elem_id !== void 0)
    $$bindings.elem_id(elem_id);
  if ($$props.elem_classes === void 0 && $$bindings.elem_classes && elem_classes !== void 0)
    $$bindings.elem_classes(elem_classes);
  if ($$props.visible === void 0 && $$bindings.visible && visible !== void 0)
    $$bindings.visible(visible);
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  if ($$props.value_is_output === void 0 && $$bindings.value_is_output && value_is_output !== void 0)
    $$bindings.value_is_output(value_is_output);
  if ($$props.col_count === void 0 && $$bindings.col_count && col_count !== void 0)
    $$bindings.col_count(col_count);
  if ($$props.row_count === void 0 && $$bindings.row_count && row_count !== void 0)
    $$bindings.row_count(row_count);
  if ($$props.label === void 0 && $$bindings.label && label !== void 0)
    $$bindings.label(label);
  if ($$props.show_label === void 0 && $$bindings.show_label && show_label !== void 0)
    $$bindings.show_label(show_label);
  if ($$props.wrap === void 0 && $$bindings.wrap && wrap !== void 0)
    $$bindings.wrap(wrap);
  if ($$props.datatype === void 0 && $$bindings.datatype && datatype !== void 0)
    $$bindings.datatype(datatype);
  if ($$props.scale === void 0 && $$bindings.scale && scale !== void 0)
    $$bindings.scale(scale);
  if ($$props.min_width === void 0 && $$bindings.min_width && min_width !== void 0)
    $$bindings.min_width(min_width);
  if ($$props.root === void 0 && $$bindings.root && root !== void 0)
    $$bindings.root(root);
  if ($$props.line_breaks === void 0 && $$bindings.line_breaks && line_breaks !== void 0)
    $$bindings.line_breaks(line_breaks);
  if ($$props.column_widths === void 0 && $$bindings.column_widths && column_widths !== void 0)
    $$bindings.column_widths(column_widths);
  if ($$props.gradio === void 0 && $$bindings.gradio && gradio !== void 0)
    $$bindings.gradio(gradio);
  if ($$props.latex_delimiters === void 0 && $$bindings.latex_delimiters && latex_delimiters !== void 0)
    $$bindings.latex_delimiters(latex_delimiters);
  if ($$props.max_height === void 0 && $$bindings.max_height && max_height !== void 0)
    $$bindings.max_height(max_height);
  if ($$props.loading_status === void 0 && $$bindings.loading_status && loading_status !== void 0)
    $$bindings.loading_status(loading_status);
  if ($$props.interactive === void 0 && $$bindings.interactive && interactive !== void 0)
    $$bindings.interactive(interactive);
  if ($$props.show_fullscreen_button === void 0 && $$bindings.show_fullscreen_button && show_fullscreen_button !== void 0)
    $$bindings.show_fullscreen_button(show_fullscreen_button);
  if ($$props.max_chars === void 0 && $$bindings.max_chars && max_chars !== void 0)
    $$bindings.max_chars(max_chars);
  if ($$props.show_copy_button === void 0 && $$bindings.show_copy_button && show_copy_button !== void 0)
    $$bindings.show_copy_button(show_copy_button);
  if ($$props.show_row_numbers === void 0 && $$bindings.show_row_numbers && show_row_numbers !== void 0)
    $$bindings.show_row_numbers(show_row_numbers);
  if ($$props.show_search === void 0 && $$bindings.show_search && show_search !== void 0)
    $$bindings.show_search(show_search);
  if ($$props.pinned_columns === void 0 && $$bindings.pinned_columns && pinned_columns !== void 0)
    $$bindings.pinned_columns(pinned_columns);
  let $$settled;
  let $$rendered;
  let previous_head = $$result.head;
  do {
    $$settled = true;
    $$result.head = previous_head;
    filtered_cell_values = null;
    _headers = [...value.headers || headers];
    display_value = value?.metadata?.display_value ? [...value?.metadata?.display_value] : null;
    styling = !interactive && value?.metadata?.styling ? [...value?.metadata?.styling] : null;
    $$rendered = `   ${validate_component(Block, "Block").$$render(
      $$result,
      {
        visible,
        padding: false,
        elem_id,
        elem_classes,
        container: false,
        scale,
        min_width,
        overflow_behavior: "visible"
      },
      {},
      {
        default: () => {
          return `${validate_component(Static, "StatusTracker").$$render($$result, Object.assign({}, { autoscroll: gradio.autoscroll }, { i18n: gradio.i18n }, loading_status), {}, {})} ${validate_component(Table$1, "Table").$$render(
            $$result,
            {
              root,
              label,
              show_label,
              row_count,
              col_count,
              values: filtered_cell_values || value.data,
              display_value,
              styling,
              headers: _headers,
              wrap,
              datatype,
              latex_delimiters,
              editable: interactive,
              max_height,
              i18n: gradio.i18n,
              line_breaks,
              column_widths,
              upload: (...args) => gradio.client.upload(...args),
              stream_handler: (...args) => gradio.client.stream(...args),
              show_fullscreen_button,
              max_chars,
              show_copy_button,
              show_row_numbers,
              show_search,
              pinned_columns,
              value_is_output
            },
            {
              value_is_output: ($$value) => {
                value_is_output = $$value;
                $$settled = false;
              }
            },
            {}
          )}`;
        }
      }
    )}`;
  } while (!$$settled);
  return $$rendered;
});

export { Table$1 as BaseDataFrame, Index as default };
//# sourceMappingURL=Index57-DNi8bVNk.js.map
