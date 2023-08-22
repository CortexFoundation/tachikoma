from . import trace

def quantize(expr, params, model_name):
    tr = Trace.from_expr(expr, params, model_name)
    fuse_tr = tr.checkpoint_transform(
            fuse.FuseTupleGetItem.apply(),
            fuse.FuseBatchNorm.apply(),
            fuse.FuseAvgPool2D.apply(),
            tr_name = "fuse",
            # force=True,
            )
    calib_tr = fuse_tr.checkpoint_transform(
            Calibrator.apply(random_config={
                "enabled": True,
                "absmax": 1.0, }),
            print_bf=True, print_af=True,
    )
    dt_tr = calib_tr.checkpoint_transform(
            SymmetricMinMaxSampling.apply(),
            slm.SymmetricLinearDiscretor.apply(),
            )
    dt_tr: Trace = dt_tr.checkpoint_transform(
            Quantizer.apply(),
            # force=True,
    )
    sim_tr = dt_tr.checkpoint_transform(
            Simulator.apply(
                with_clip=False, with_round=False),
            force=True,
            )
    return sim_tr
