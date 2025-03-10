import { SvelteComponent } from "svelte";
import type { I18nFormatter } from "@gradio/utils";
declare const __propDef: {
    props: {
        direction?: (("asc" | "des") | null) | undefined;
        i18n: I18nFormatter;
    };
    events: {
        sort: CustomEvent<"asc" | "des">;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type SortIconProps = typeof __propDef.props;
export type SortIconEvents = typeof __propDef.events;
export type SortIconSlots = typeof __propDef.slots;
export default class SortIcon extends SvelteComponent<SortIconProps, SortIconEvents, SortIconSlots> {
}
export {};
