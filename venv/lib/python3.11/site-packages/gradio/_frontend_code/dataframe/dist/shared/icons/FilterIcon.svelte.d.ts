import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        [x: string]: never;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type FilterIconProps = typeof __propDef.props;
export type FilterIconEvents = typeof __propDef.events;
export type FilterIconSlots = typeof __propDef.slots;
export default class FilterIcon extends SvelteComponent<FilterIconProps, FilterIconEvents, FilterIconSlots> {
}
export {};
